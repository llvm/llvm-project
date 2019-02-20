//===--- FillInMissingMethodStubsFromAbstractClasses.cpp -  ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the "Add missing abstract class method overrides" refactoring
// operation.
//
//===----------------------------------------------------------------------===//

#include "RefactoringOperations.h"
#include "SourceLocationUtilities.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/DenseSet.h"

using namespace clang;
using namespace clang::tooling;

namespace {

class FillInMissingMethodStubsFromAbstractClassesOperation
    : public RefactoringOperation {
public:
  FillInMissingMethodStubsFromAbstractClassesOperation(
      const CXXRecordDecl *Class)
      : Class(Class) {}

  const Decl *getTransformedDecl() const override { return Class; }

  llvm::Expected<RefactoringResult> perform(ASTContext &Context, const Preprocessor &ThePreprocessor,
          const RefactoringOptionSet &Options,
          unsigned SelectedCandidateIndex) override;

  const CXXRecordDecl *Class;
};

} // end anonymous namespace

static bool hasAbstractBases(const CXXRecordDecl *Class) {
  for (const CXXBaseSpecifier &Base : Class->bases()) {
    if (const auto *RD = Base.getType()->getAsCXXRecordDecl()) {
      if (RD->isAbstract())
        return true;
    }
  }
  return false;
}

RefactoringOperationResult
clang::tooling::initiateFillInMissingMethodStubsFromAbstractClassesOperation(
    ASTSlice &Slice, ASTContext &Context, SourceLocation Location,
    SourceRange SelectionRange, bool CreateOperation) {
  auto SelectedDecl = Slice.innermostSelectedDecl(
      llvm::makeArrayRef(Decl::CXXRecord), ASTSlice::InnermostDeclOnly);
  if (!SelectedDecl)
    return None;
  const auto *Class = cast<CXXRecordDecl>(SelectedDecl->getDecl());
  if (Class->isUnion() || !Class->isThisDeclarationADefinition())
    return None;
  if (!hasAbstractBases(Class))
    return RefactoringOperationResult("The class has no abstract bases");
  if (!Class->isDependentType() && !Class->isAbstract())
    return RefactoringOperationResult(
        "The class has no missing abstract class methods");

  RefactoringOperationResult Result;
  Result.Initiated = true;
  if (!CreateOperation)
    return Result;
  auto Operation =
      llvm::make_unique<FillInMissingMethodStubsFromAbstractClassesOperation>(
          Class);
  Result.RefactoringOp = std::move(Operation);
  return Result;
}

namespace {

class PureMethodSet {
  llvm::DenseMap<const CXXMethodDecl *, int> Methods;

  void addPureMethodsFromAbstractClasses(const CXXRecordDecl *Class,
                                         int &Priority) {
    for (const CXXBaseSpecifier &Base : Class->bases()) {
      const auto *RD = Base.getType()->getAsCXXRecordDecl();
      if (!RD || !RD->isAbstract())
        continue;
      for (const CXXMethodDecl *M : RD->methods()) {
        if (M->isPure())
          Methods.insert(std::make_pair(M->getCanonicalDecl(), Priority++));
      }
      addPureMethodsFromAbstractClasses(RD, Priority);
    }
  }

  void addPureMethodsFromAbstractClasses(const CXXRecordDecl *Class) {
    int Priority = 0;
    addPureMethodsFromAbstractClasses(Class, Priority);
  }

  void subtractImplementedPureMethods(const CXXRecordDecl *Class) {
    for (const CXXMethodDecl *M : Class->methods()) {
      if (!M->isVirtual() || M->isPure())
        continue;
      for (const CXXMethodDecl *OM : M->overridden_methods()) {
        OM = OM->getCanonicalDecl();
        if (OM->isPure())
          Methods.erase(OM);
      }
    }
    for (const CXXBaseSpecifier &Base : Class->bases()) {
      const auto *RD = Base.getType()->getAsCXXRecordDecl();
      if (!RD || !RD->isAbstract())
        continue;
      subtractImplementedPureMethods(RD);
    }
  }

public:
  static std::vector<const CXXMethodDecl *>
  gatherMissingMethods(const CXXRecordDecl *Class) {
    PureMethodSet MethodSet;
    MethodSet.addPureMethodsFromAbstractClasses(Class);
    MethodSet.subtractImplementedPureMethods(Class);
    // Sort the missing methods. That will place methods from the same abstract
    // class together in the order in which they were declared.
    struct MethodInfo {
      const CXXMethodDecl *M;
      int Priority;
    };
    std::vector<MethodInfo> MissingMethods;
    for (const auto &M : MethodSet.Methods)
      MissingMethods.push_back({M.first, M.second});
    std::sort(MissingMethods.begin(), MissingMethods.end(),
              [](const MethodInfo &LHS, const MethodInfo &RHS) {
                return LHS.Priority < RHS.Priority;
              });
    std::vector<const CXXMethodDecl *> Result;
    Result.reserve(MissingMethods.size());
    for (const auto &M : MissingMethods)
      Result.push_back(M.M);
    return Result;
  }
};

} // end anonymous namespace

static SourceLocation findInsertionLocationForMethodsFromAbstractClass(
    const CXXRecordDecl *AbstractClass, const CXXRecordDecl *Class,
    const SourceManager &SM, const LangOptions &LangOpts) {
  SourceLocation Loc;
  for (const CXXMethodDecl *M : Class->methods()) {
    if (!M->isVirtual() || M->isPure() || M->isImplicit())
      continue;
    for (const CXXMethodDecl *OM : M->overridden_methods()) {
      OM = OM->getCanonicalDecl();
      if (OM->getLexicalDeclContext() == AbstractClass) {
        SourceLocation EndLoc = M->getEndLoc();
        if (EndLoc.isMacroID())
          EndLoc = SM.getExpansionRange(EndLoc).getEnd();
        if (Loc.isInvalid())
          Loc = EndLoc;
        else if (SM.isBeforeInTranslationUnit(Loc, EndLoc))
          Loc = EndLoc;
        break;
      }
    }
  }
  if (Loc.isInvalid())
    return Loc;
  return getLastLineLocationUnlessItHasOtherTokens(Loc, SM, LangOpts);
}

/// Returns true if the given \p Class implements the majority of declared
/// methods in the class itself.
static bool shouldImplementMethodsInClass(const CXXRecordDecl *Class) {
  // Check if this class implements the methods in the class itself.
  unsigned NumMethods = 0, NumImplementedMethods = 0;
  for (const CXXMethodDecl *M : Class->methods()) {
    if (M->isImplicit())
      continue;
    // Only look at methods/operators.
    if (isa<CXXConstructorDecl>(M) || isa<CXXDestructorDecl>(M))
      continue;
    ++NumMethods;
    if (M->hasBody())
      ++NumImplementedMethods;
  }
  if (!NumMethods)
    return false;
  // Use the following arbitrary heuristic:
  // If the number of method declarations is less than 4, then all of the
  // methods must have bodies. Otherwise, at least 75% of the methods must
  // have bodies.
  return NumMethods < 4
             ? NumMethods == NumImplementedMethods
             : float(NumImplementedMethods) / float(NumMethods) > 0.75;
}

llvm::Expected<RefactoringResult>
FillInMissingMethodStubsFromAbstractClassesOperation::perform(
    ASTContext &Context, const Preprocessor &ThePreprocessor,
    const RefactoringOptionSet &Options, unsigned SelectedCandidateIndex) {
  std::vector<RefactoringReplacement> Replacements;

  std::vector<const CXXMethodDecl *> MissingMethods =
      PureMethodSet::gatherMissingMethods(Class);

  bool GenerateBodyDummies = shouldImplementMethodsInClass(Class);

  PrintingPolicy PP = Context.getPrintingPolicy();
  PP.PolishForDeclaration = true;
  PP.SuppressStrongLifetime = true;
  PP.SuppressLifetimeQualifiers = true;
  PP.SuppressUnwrittenScope = true;

  std::string EndInsertionOSStr;
  llvm::raw_string_ostream EndInsertionOS(EndInsertionOSStr);

  std::string InsertionGroupOSStr;
  llvm::raw_string_ostream InsertionGroupOS(InsertionGroupOSStr);

  SourceLocation InsertionLoc = Class->getEndLoc();
  const CXXRecordDecl *CurrentAbstractClass = nullptr;
  SourceLocation CurrentGroupInsertionLoc;
  for (const auto &I : llvm::enumerate(MissingMethods)) {
    const CXXMethodDecl *Method = I.value();
    const CXXRecordDecl *AbstractClass = Method->getParent();
    if (CurrentAbstractClass != AbstractClass) {
      if (!InsertionGroupOS.str().empty()) {
        assert(CurrentGroupInsertionLoc.isValid());
        Replacements.emplace_back(
            SourceRange(CurrentGroupInsertionLoc, CurrentGroupInsertionLoc),
            InsertionGroupOS.str());
      }
      InsertionGroupOSStr.clear();
      CurrentAbstractClass = AbstractClass;
      CurrentGroupInsertionLoc =
          findInsertionLocationForMethodsFromAbstractClass(
              CurrentAbstractClass, Class, Context.getSourceManager(),
              Context.getLangOpts());
    }
    bool IsInsertingAfterRelatedMethods = CurrentGroupInsertionLoc.isValid();
    raw_ostream &OS =
        IsInsertingAfterRelatedMethods ? InsertionGroupOS : EndInsertionOS;

    if (IsInsertingAfterRelatedMethods && InsertionGroupOS.str().empty())
      OS << "\n\n";
    // Print the method without the 'virtual' specifier and the pure '= 0'
    // annotation.
    auto *MD = const_cast<CXXMethodDecl *>(Method);
    bool IsVirtual = MD->isVirtualAsWritten();
    MD->setVirtualAsWritten(false);
    bool IsPure = MD->isPure();
    MD->setPure(false);
    MD->print(OS, PP);
    MD->setVirtualAsWritten(IsVirtual);
    MD->setPure(IsPure);

    OS << " override";
    if (GenerateBodyDummies)
      OS << " { \n  <#code#>\n}\n";
    else
      OS << ";\n";
    // Avoid an additional newline for the last method in an insertion group.
    if (IsInsertingAfterRelatedMethods) {
      const CXXRecordDecl *NextAbstractClass =
          (I.index() + 1) != MissingMethods.size()
              ? MissingMethods[I.index() + 1]->getParent()
              : nullptr;
      if (NextAbstractClass == CurrentAbstractClass)
        OS << "\n";
    } else
      OS << "\n";
  }
  if (!InsertionGroupOS.str().empty()) {
    assert(CurrentGroupInsertionLoc.isValid());
    Replacements.emplace_back(
        SourceRange(CurrentGroupInsertionLoc, CurrentGroupInsertionLoc),
        InsertionGroupOS.str());
  }
  if (!EndInsertionOS.str().empty())
    Replacements.emplace_back(SourceRange(InsertionLoc, InsertionLoc),
                              EndInsertionOS.str());

  return std::move(Replacements);
}
