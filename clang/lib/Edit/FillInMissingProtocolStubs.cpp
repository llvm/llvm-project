//===--- FillInMissingProtocolStubs.cpp -  --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the "Add methods from protocol(s)" refactoring operation.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NSAPI.h"
#include "clang/Edit/RefactoringFixits.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>

using namespace clang;
using namespace edit;
using namespace fillInMissingProtocolStubs;

// FIXME: This is duplicated with the refactoring lib.
static bool areOnSameLine(SourceLocation Loc1, SourceLocation Loc2,
                          const SourceManager &SM) {
  return !Loc1.isMacroID() && !Loc2.isMacroID() &&
         SM.getSpellingLineNumber(Loc1) == SM.getSpellingLineNumber(Loc2);
}

static bool isSemicolonAtLocation(SourceLocation TokenLoc,
                                  const SourceManager &SM,
                                  const LangOptions &LangOpts) {
  return Lexer::getSourceText(
             CharSourceRange::getTokenRange(TokenLoc, TokenLoc), SM,
             LangOpts) == ";";
}

static SourceLocation getLocationOfPrecedingToken(SourceLocation Loc,
                                                  const SourceManager &SM,
                                                  const LangOptions &LangOpts) {
  SourceLocation Result = Loc;
  if (Result.isMacroID())
    Result = SM.getExpansionLoc(Result);
  FileID FID = SM.getFileID(Result);
  SourceLocation StartOfFile = SM.getLocForStartOfFile(FID);
  if (Loc == StartOfFile)
    return SourceLocation();
  return Lexer::GetBeginningOfToken(Result.getLocWithOffset(-1), SM, LangOpts);
}

static SourceLocation
getLastLineLocationUnlessItHasOtherTokens(SourceLocation SpellingLoc,
                                          const SourceManager &SM,
                                          const LangOptions &LangOpts) {
  assert(!SpellingLoc.isMacroID() && "Expecting a spelling location");
  SourceLocation NextTokenLoc =
      Lexer::findNextTokenLocationAfterTokenAt(SpellingLoc, SM, LangOpts);
  if (NextTokenLoc.isValid()) {
    bool IsSameLine = areOnSameLine(SpellingLoc, NextTokenLoc, SM);
    if (IsSameLine) {
      // Could be a ';' on the same line, so try looking after the ';'
      if (isSemicolonAtLocation(NextTokenLoc, SM, LangOpts))
        return getLastLineLocationUnlessItHasOtherTokens(NextTokenLoc, SM,
                                                         LangOpts);
    } else {
      SourceLocation LastLoc = SM.translateLineCol(
          SM.getFileID(SpellingLoc), SM.getSpellingLineNumber(SpellingLoc),
          std::numeric_limits<unsigned>::max());
      if (LastLoc.isValid())
        return LastLoc;
    }
  }
  return Lexer::getLocForEndOfToken(SpellingLoc, 0, SM, LangOpts);
}

namespace {

struct ProtocolInfo {
  /// The lower the priority, the more important this protocol is considered to
  /// be. Typically protocols from the class have lower priority than protocols
  /// from superclasses.
  int Priority;
};

using ProtocolMapTy = llvm::DenseMap<const ObjCProtocolDecl *, ProtocolInfo>;

/// Contains the set of methods from all the protocols that the class conforms
/// to.
class MethodSet {
public:
  struct MethodInfo {
    const ObjCMethodDecl *M;
    const ObjCProtocolDecl *P;
    int ProtocolPriority;
    enum MethodPresenceKind { IsDeclared = 0x1, IsImplemented = 0x2 };
    unsigned PresenceKind = 0;
    const ObjCMethodDecl *DeclaredOrImplementedMethod = nullptr;

    MethodInfo(const ObjCMethodDecl *M, const ObjCProtocolDecl *P,
               int ProtocolPriority)
        : M(M), P(P), ProtocolPriority(ProtocolPriority) {}

    bool isRequired() const {
      return M->getImplementationControl() == ObjCMethodDecl::Required;
    }
    void markAs(MethodPresenceKind Kind) { PresenceKind |= Kind; }
    bool is(MethodPresenceKind Kind) const {
      return (PresenceKind & Kind) == Kind;
    }
  };

private:
  llvm::DenseMap<Selector, MethodInfo> InstanceMethods;
  llvm::DenseMap<Selector, MethodInfo> ClassMethods;

  void markMethodsFrom(const ObjCContainerDecl *Container,
                       MethodInfo::MethodPresenceKind Kind) {
    for (const ObjCMethodDecl *M : Container->methods()) {
      auto &Map = M->isInstanceMethod() ? InstanceMethods : ClassMethods;
      auto It = Map.find(M->getSelector());
      if (It != Map.end()) {
        It->second.markAs(Kind);
        if (!It->second.DeclaredOrImplementedMethod)
          It->second.DeclaredOrImplementedMethod = M;
      }
    }
  }

public:
  MethodSet() {}
  MethodSet(MethodSet &&Other) = default;
  MethodSet &operator=(MethodSet &&Other) = default;

  void gatherMethodsFrom(const ObjCProtocolDecl *P, int Priority) {
    for (const ObjCMethodDecl *M : P->methods()) {
      if (M->isImplicit())
        continue;
      AvailabilityResult Availability = M->getAvailability();
      // Methods that are unavailable or not yet introduced are not considered
      // to be required.
      if (Availability == AR_NotYetIntroduced || Availability == AR_Unavailable)
        continue;
      auto &Map = M->isInstanceMethod() ? InstanceMethods : ClassMethods;
      Map.insert(std::make_pair(M->getSelector(), MethodInfo(M, P, Priority)));
    }
  }

  void markImplementedMethods(const ObjCContainerDecl *Container) {
    assert(isa<ObjCImplDecl>(Container) && "Not an implementation container");
    markMethodsFrom(Container, MethodInfo::IsImplemented);
    if (const auto *ID = dyn_cast<ObjCImplementationDecl>(Container)) {
      const auto *I = ID->getClassInterface();
      // Mark declarations from super-classes as implemented to prevent
      // redundant implementations.
      while ((I = I->getSuperClass()))
        markMethodsFrom(I, MethodInfo::IsImplemented);
    }
  }

  void markDeclaredMethods(const ObjCContainerDecl *Container) {
    assert(!isa<ObjCImplDecl>(Container) && "Not an interface container");
    markMethodsFrom(Container, MethodInfo::IsDeclared);
    // Mark declarations from super-classes as declared to prevent redundant
    // declarations.
    if (const auto *I = dyn_cast<ObjCInterfaceDecl>(Container)) {
      while ((I = I->getSuperClass()))
        markMethodsFrom(I, MethodInfo::IsDeclared);
    }
  }

  /// Returns true if the given container has missing @required method stubs.
  ///
  /// For @interfaces, this method returns true when the interface is missing
  /// a declaration for any @required method in all of the protocols.
  /// For @implementations, this method returns true when the implementation is
  /// missing an implementation of any @required method in all of the protocols.
  bool hasMissingRequiredMethodStubs(const ObjCContainerDecl *Container) {
    MethodInfo::MethodPresenceKind Kind = isa<ObjCImplDecl>(Container)
                                              ? MethodInfo::IsImplemented
                                              : MethodInfo::IsDeclared;
    for (const auto &I : InstanceMethods) {
      if (!I.second.isRequired())
        continue;
      if (!I.second.is(Kind))
        return true;
    }
    for (const auto &I : ClassMethods) {
      if (!I.second.isRequired())
        continue;
      if (!I.second.is(Kind))
        return true;
    }
    return false;
  }

  std::vector<MethodInfo>
  getMissingRequiredMethods(const ObjCContainerDecl *Container) {
    MethodInfo::MethodPresenceKind Kind = isa<ObjCImplDecl>(Container)
                                              ? MethodInfo::IsImplemented
                                              : MethodInfo::IsDeclared;
    std::vector<MethodInfo> Results;
    for (const auto &I : InstanceMethods) {
      if (!I.second.isRequired())
        continue;
      if (!I.second.is(Kind))
        Results.push_back(I.second);
    }
    for (const auto &I : ClassMethods) {
      if (!I.second.isRequired())
        continue;
      if (!I.second.is(Kind))
        Results.push_back(I.second);
    }
    return Results;
  }

  SourceLocation findLocationForInsertionForMethodsFromProtocol(
      const ObjCProtocolDecl *P, const ObjCContainerDecl *Container,
      const SourceManager &SM, const LangOptions &LangOpts) {
    MethodInfo::MethodPresenceKind Kind = isa<ObjCImplDecl>(Container)
                                              ? MethodInfo::IsImplemented
                                              : MethodInfo::IsDeclared;
    llvm::SmallVector<const ObjCMethodDecl *, 4> MethodsFromProtocolInContainer;
    for (const ObjCMethodDecl *M : P->methods()) {
      if (M->isImplicit())
        continue;
      const auto &Map = M->isInstanceMethod() ? InstanceMethods : ClassMethods;
      auto It = Map.find(M->getSelector());
      if (It == Map.end())
        continue;
      if (!It->second.is(Kind))
        continue;
      const ObjCMethodDecl *ContainerMethod =
          It->second.DeclaredOrImplementedMethod;
      // Ignore method declarations from superclasses.
      if (ContainerMethod->getLexicalDeclContext() != Container)
        continue;
      // This is a method from the given protocol that either declared or
      // implemented in the container.
      MethodsFromProtocolInContainer.push_back(ContainerMethod);
    }
    // Find the appropriate source locations by looking
    if (MethodsFromProtocolInContainer.empty())
      return SourceLocation();
    SourceLocation Loc = MethodsFromProtocolInContainer[0]->getEndLoc();
    if (Loc.isMacroID())
      Loc = SM.getExpansionRange(Loc).getEnd();
    for (const ObjCMethodDecl *M :
         makeArrayRef(MethodsFromProtocolInContainer).drop_front()) {
      SourceLocation EndLoc = M->getEndLoc();
      if (EndLoc.isMacroID())
        EndLoc = SM.getExpansionRange(EndLoc).getEnd();
      if (SM.isBeforeInTranslationUnit(Loc, EndLoc))
        Loc = EndLoc;
    }
    return getLastLineLocationUnlessItHasOtherTokens(Loc, SM, LangOpts);
  }
};

} // end anonymous namespace

namespace clang {
namespace edit {
namespace fillInMissingProtocolStubs {

class FillInMissingProtocolStubsImpl {
public:
  const ObjCContainerDecl *Container;
  MethodSet Methods;
};

} // end namespace fillInMissingProtocolStubsImpl
} // end namespace edit
} // end namespace clang

static void gatherProtocols(
    llvm::iterator_range<ObjCList<ObjCProtocolDecl>::iterator> Protocols,
    NSAPI &API, ProtocolMapTy &Result, int &Priority) {
  for (const ObjCProtocolDecl *P : Protocols) {
    // Ignore the 'NSObject' protocol.
    if (API.getNSClassId(NSAPI::ClassId_NSObject) == P->getIdentifier())
      continue;
    gatherProtocols(P->protocols(), API, Result, Priority);
    Result.insert(std::make_pair(P, ProtocolInfo{Priority++}));
  }
}

static ProtocolMapTy
gatherSuitableClassProtocols(const ObjCInterfaceDecl *I,
                             const ObjCContainerDecl *Container, NSAPI &API) {
  ProtocolMapTy Result;
  // The class of interest should use the protocols from extensions when the
  // operation is initiated from the @implementation / extension.
  auto ClassProtocols =
      Container == I ? I->protocols() : I->all_referenced_protocols();
  int Priority = 0;
  gatherProtocols(ClassProtocols, API, Result, Priority);
  while ((I = I->getSuperClass()))
    gatherProtocols(I->protocols(), API, Result, Priority);
  return Result;
}

static const ObjCContainerDecl *
getInterfaceOrCategory(const ObjCContainerDecl *Container) {
  if (const auto *Impl = dyn_cast<ObjCImplementationDecl>(Container))
    return Impl->getClassInterface();
  if (const auto *CategoryImpl = dyn_cast<ObjCCategoryImplDecl>(Container))
    return CategoryImpl->getCategoryDecl();
  return Container;
}

static bool initiate(FillInMissingProtocolStubsImpl &Dest, ASTContext &Context,
                     const ObjCContainerDecl *Container) {
  const ObjCContainerDecl *ContainerProtocolSource =
      getInterfaceOrCategory(Container);
  if (!ContainerProtocolSource)
    return false;

  // The protocols that are specified in the @interface and/or in the
  // superclasses.
  ProtocolMapTy Protocols;
  NSAPI API(Context);
  if (const auto *I = dyn_cast<ObjCInterfaceDecl>(ContainerProtocolSource)) {
    if (!I->hasDefinition())
      return false;
    Protocols = gatherSuitableClassProtocols(I, Container, API);
    if (Protocols.empty())
      return false;
  } else if (const auto *I =
                 dyn_cast<ObjCCategoryDecl>(ContainerProtocolSource)) {
    int Priority = 0;
    gatherProtocols(I->protocols(), API, Protocols, Priority);
    if (Protocols.empty())
      return false;
  }

  // Check if there are missing @required methods.
  for (const auto &P : Protocols)
    Dest.Methods.gatherMethodsFrom(P.first, P.second.Priority);
  if (isa<ObjCImplDecl>(Container))
    Dest.Methods.markImplementedMethods(Container);
  else
    Dest.Methods.markDeclaredMethods(Container);

  Dest.Container = Container;
  return true;
}

FillInMissingProtocolStubs::FillInMissingProtocolStubs() {}
FillInMissingProtocolStubs::~FillInMissingProtocolStubs() {}
FillInMissingProtocolStubs::FillInMissingProtocolStubs(
    FillInMissingProtocolStubs &&Other)
    : Impl(std::move(Other.Impl)) {}
FillInMissingProtocolStubs &FillInMissingProtocolStubs::
operator=(FillInMissingProtocolStubs &&Other) {
  Impl = std::move(Other.Impl);
  return *this;
}

bool FillInMissingProtocolStubs::initiate(ASTContext &Context,
                                          const ObjCContainerDecl *Container) {
  Impl = llvm::make_unique<FillInMissingProtocolStubsImpl>();
  if (!::initiate(*Impl, Context, Container))
    return true;
  return false;
}

bool FillInMissingProtocolStubs::hasMissingRequiredMethodStubs() {
  return Impl->Methods.hasMissingRequiredMethodStubs(Impl->Container);
}

static void perform(MethodSet &Methods, const ObjCContainerDecl *Container,
                    ASTContext &Context,
                    llvm::function_ref<void(const FixItHint &)> Consumer) {
  auto MissingMethods = Methods.getMissingRequiredMethods(Container);
  // Sort the methods by grouping them into protocol clusters and then sorting
  // them alphabetically within the same protocol.
  std::sort(MissingMethods.begin(), MissingMethods.end(),
            [](const MethodSet::MethodInfo &A, const MethodSet::MethodInfo &B) {
              if (A.ProtocolPriority == B.ProtocolPriority)
                return A.M->getSelector().getAsString() <
                       B.M->getSelector().getAsString();
              assert(A.P != B.P && "Same protocols should have same priority");
              return A.ProtocolPriority < B.ProtocolPriority;
            });

  SourceLocation InsertionLoc =
      isa<ObjCImplDecl>(Container)
          ? Container->getEndLoc()
          : getLocationOfPrecedingToken(Container->getEndLoc(),
                                        Context.getSourceManager(),
                                        Context.getLangOpts());
  if (InsertionLoc.isInvalid())
    InsertionLoc = Container->getEndLoc();

  PrintingPolicy PP = Context.getPrintingPolicy();
  PP.PolishForDeclaration = true;
  PP.SuppressStrongLifetime = true;
  PP.SuppressLifetimeQualifiers = true;
  PP.SuppressUnwrittenScope = true;

  std::string EndInsertionOSStr;
  llvm::raw_string_ostream EndInsertionOS(EndInsertionOSStr);

  std::string InsertionGroupStr;
  llvm::raw_string_ostream InsertionGroupOS(InsertionGroupStr);

  const ObjCProtocolDecl *CurrentProtocol = nullptr;
  SourceLocation CurrentProtocolInsertionLoc;
  bool IsImplementation = isa<ObjCImplDecl>(Container);
  for (const auto &Method : MissingMethods) {
    const ObjCProtocolDecl *P = Method.P;
    if (CurrentProtocol != P) {
      if (!InsertionGroupOS.str().empty()) {
        assert(CurrentProtocolInsertionLoc.isValid());
        Consumer(FixItHint::CreateInsertion(CurrentProtocolInsertionLoc,
                                            InsertionGroupOS.str()));
      }
      InsertionGroupStr.clear();
      CurrentProtocol = P;
      CurrentProtocolInsertionLoc =
          Methods.findLocationForInsertionForMethodsFromProtocol(
              P, Container, Context.getSourceManager(), Context.getLangOpts());
    }
    bool IsInsertingAfterRelatedMethods = CurrentProtocolInsertionLoc.isValid();
    raw_ostream &OS =
        IsInsertingAfterRelatedMethods ? InsertionGroupOS : EndInsertionOS;

    std::string MethodDeclStr;
    llvm::raw_string_ostream MethodOS(MethodDeclStr);
    Method.M->print(MethodOS, PP);
    if (IsInsertingAfterRelatedMethods)
      OS << "\n\n";
    OS << StringRef(MethodOS.str()).drop_back(); // Drop the ';'
    if (IsImplementation)
      OS << " { \n  <#code#>\n}\n";
    else
      OS << ";\n";
    if (!IsInsertingAfterRelatedMethods)
      OS << "\n";
  }
  if (!InsertionGroupOS.str().empty()) {
    assert(CurrentProtocolInsertionLoc.isValid());
    Consumer(FixItHint::CreateInsertion(CurrentProtocolInsertionLoc,
                                        InsertionGroupOS.str()));
  }
  if (!EndInsertionOS.str().empty())
    Consumer(FixItHint::CreateInsertion(InsertionLoc, EndInsertionOS.str()));
}

void FillInMissingProtocolStubs::perform(
    ASTContext &Context, llvm::function_ref<void(const FixItHint &)> Consumer) {
  ::perform(Impl->Methods, Impl->Container, Context, Consumer);
}

void fillInMissingProtocolStubs::addMissingProtocolStubs(
    ASTContext &Context, const ObjCContainerDecl *Container,
    llvm::function_ref<void(const FixItHint &)> Consumer) {
  FillInMissingProtocolStubsImpl Impl;
  if (initiate(Impl, Context, Container))
    perform(Impl.Methods, Impl.Container, Context, Consumer);
}
