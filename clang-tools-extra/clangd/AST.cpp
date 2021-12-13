//===--- AST.cpp - Utility AST functions  -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"

#include "FindTarget.h"
#include "SourceCode.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace clang {
namespace clangd {

namespace {
llvm::Optional<llvm::ArrayRef<TemplateArgumentLoc>>
getTemplateSpecializationArgLocs(const NamedDecl &ND) {
  if (auto *Func = llvm::dyn_cast<FunctionDecl>(&ND)) {
    if (const ASTTemplateArgumentListInfo *Args =
            Func->getTemplateSpecializationArgsAsWritten())
      return Args->arguments();
  } else if (auto *Cls =
                 llvm::dyn_cast<ClassTemplatePartialSpecializationDecl>(&ND)) {
    if (auto *Args = Cls->getTemplateArgsAsWritten())
      return Args->arguments();
  } else if (auto *Var =
                 llvm::dyn_cast<VarTemplatePartialSpecializationDecl>(&ND)) {
    if (auto *Args = Var->getTemplateArgsAsWritten())
      return Args->arguments();
  } else if (auto *Var = llvm::dyn_cast<VarTemplateSpecializationDecl>(&ND))
    return Var->getTemplateArgsInfo().arguments();
  // We return None for ClassTemplateSpecializationDecls because it does not
  // contain TemplateArgumentLoc information.
  return llvm::None;
}

template <class T>
bool isTemplateSpecializationKind(const NamedDecl *D,
                                  TemplateSpecializationKind Kind) {
  if (const auto *TD = dyn_cast<T>(D))
    return TD->getTemplateSpecializationKind() == Kind;
  return false;
}

bool isTemplateSpecializationKind(const NamedDecl *D,
                                  TemplateSpecializationKind Kind) {
  return isTemplateSpecializationKind<FunctionDecl>(D, Kind) ||
         isTemplateSpecializationKind<CXXRecordDecl>(D, Kind) ||
         isTemplateSpecializationKind<VarDecl>(D, Kind);
}

// Store all UsingDirectiveDecls in parent contexts of DestContext, that were
// introduced before InsertionPoint.
llvm::DenseSet<const NamespaceDecl *>
getUsingNamespaceDirectives(const DeclContext *DestContext,
                            SourceLocation Until) {
  const auto &SM = DestContext->getParentASTContext().getSourceManager();
  llvm::DenseSet<const NamespaceDecl *> VisibleNamespaceDecls;
  for (const auto *DC = DestContext; DC; DC = DC->getLookupParent()) {
    for (const auto *D : DC->decls()) {
      if (!SM.isWrittenInSameFile(D->getLocation(), Until) ||
          !SM.isBeforeInTranslationUnit(D->getLocation(), Until))
        continue;
      if (auto *UDD = llvm::dyn_cast<UsingDirectiveDecl>(D))
        VisibleNamespaceDecls.insert(
            UDD->getNominatedNamespace()->getCanonicalDecl());
    }
  }
  return VisibleNamespaceDecls;
}

// Goes over all parents of SourceContext until we find a common ancestor for
// DestContext and SourceContext. Any qualifier including and above common
// ancestor is redundant, therefore we stop at lowest common ancestor.
// In addition to that stops early whenever IsVisible returns true. This can be
// used to implement support for "using namespace" decls.
std::string
getQualification(ASTContext &Context, const DeclContext *DestContext,
                 const DeclContext *SourceContext,
                 llvm::function_ref<bool(NestedNameSpecifier *)> IsVisible) {
  std::vector<const NestedNameSpecifier *> Parents;
  bool ReachedNS = false;
  for (const DeclContext *CurContext = SourceContext; CurContext;
       CurContext = CurContext->getLookupParent()) {
    // Stop once we reach a common ancestor.
    if (CurContext->Encloses(DestContext))
      break;

    NestedNameSpecifier *NNS = nullptr;
    if (auto *TD = llvm::dyn_cast<TagDecl>(CurContext)) {
      // There can't be any more tag parents after hitting a namespace.
      assert(!ReachedNS);
      (void)ReachedNS;
      NNS = NestedNameSpecifier::Create(Context, nullptr, false,
                                        TD->getTypeForDecl());
    } else if (auto *NSD = llvm::dyn_cast<NamespaceDecl>(CurContext)) {
      ReachedNS = true;
      NNS = NestedNameSpecifier::Create(Context, nullptr, NSD);
      // Anonymous and inline namespace names are not spelled while qualifying
      // a name, so skip those.
      if (NSD->isAnonymousNamespace() || NSD->isInlineNamespace())
        continue;
    } else {
      // Other types of contexts cannot be spelled in code, just skip over
      // them.
      continue;
    }
    // Stop if this namespace is already visible at DestContext.
    if (IsVisible(NNS))
      break;

    Parents.push_back(NNS);
  }

  // Go over name-specifiers in reverse order to create necessary qualification,
  // since we stored inner-most parent first.
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  for (const auto *Parent : llvm::reverse(Parents))
    Parent->print(OS, Context.getPrintingPolicy());
  return OS.str();
}

} // namespace

bool isImplicitTemplateInstantiation(const NamedDecl *D) {
  return isTemplateSpecializationKind(D, TSK_ImplicitInstantiation);
}

bool isExplicitTemplateSpecialization(const NamedDecl *D) {
  return isTemplateSpecializationKind(D, TSK_ExplicitSpecialization);
}

bool isImplementationDetail(const Decl *D) {
  return !isSpelledInSource(D->getLocation(),
                            D->getASTContext().getSourceManager());
}

SourceLocation nameLocation(const clang::Decl &D, const SourceManager &SM) {
  auto L = D.getLocation();
  if (isSpelledInSource(L, SM))
    return SM.getSpellingLoc(L);
  return SM.getExpansionLoc(L);
}

std::string printQualifiedName(const NamedDecl &ND) {
  std::string QName;
  llvm::raw_string_ostream OS(QName);
  PrintingPolicy Policy(ND.getASTContext().getLangOpts());
  // Note that inline namespaces are treated as transparent scopes. This
  // reflects the way they're most commonly used for lookup. Ideally we'd
  // include them, but at query time it's hard to find all the inline
  // namespaces to query: the preamble doesn't have a dedicated list.
  Policy.SuppressUnwrittenScope = true;
  ND.printQualifiedName(OS, Policy);
  OS.flush();
  assert(!StringRef(QName).startswith("::"));
  return QName;
}

static bool isAnonymous(const DeclarationName &N) {
  return N.isIdentifier() && !N.getAsIdentifierInfo();
}

NestedNameSpecifierLoc getQualifierLoc(const NamedDecl &ND) {
  if (auto *V = llvm::dyn_cast<DeclaratorDecl>(&ND))
    return V->getQualifierLoc();
  if (auto *T = llvm::dyn_cast<TagDecl>(&ND))
    return T->getQualifierLoc();
  return NestedNameSpecifierLoc();
}

std::string printUsingNamespaceName(const ASTContext &Ctx,
                                    const UsingDirectiveDecl &D) {
  PrintingPolicy PP(Ctx.getLangOpts());
  std::string Name;
  llvm::raw_string_ostream Out(Name);

  if (auto *Qual = D.getQualifier())
    Qual->print(Out, PP);
  D.getNominatedNamespaceAsWritten()->printName(Out);
  return Out.str();
}

std::string printName(const ASTContext &Ctx, const NamedDecl &ND) {
  std::string Name;
  llvm::raw_string_ostream Out(Name);
  PrintingPolicy PP(Ctx.getLangOpts());
  // We don't consider a class template's args part of the constructor name.
  PP.SuppressTemplateArgsInCXXConstructors = true;

  // Handle 'using namespace'. They all have the same name - <using-directive>.
  if (auto *UD = llvm::dyn_cast<UsingDirectiveDecl>(&ND)) {
    Out << "using namespace ";
    if (auto *Qual = UD->getQualifier())
      Qual->print(Out, PP);
    UD->getNominatedNamespaceAsWritten()->printName(Out);
    return Out.str();
  }

  if (isAnonymous(ND.getDeclName())) {
    // Come up with a presentation for an anonymous entity.
    if (isa<NamespaceDecl>(ND))
      return "(anonymous namespace)";
    if (auto *Cls = llvm::dyn_cast<RecordDecl>(&ND)) {
      if (Cls->isLambda())
        return "(lambda)";
      return ("(anonymous " + Cls->getKindName() + ")").str();
    }
    if (isa<EnumDecl>(ND))
      return "(anonymous enum)";
    return "(anonymous)";
  }

  // Print nested name qualifier if it was written in the source code.
  if (auto *Qualifier = getQualifierLoc(ND).getNestedNameSpecifier())
    Qualifier->print(Out, PP);
  // Print the name itself.
  ND.getDeclName().print(Out, PP);
  // Print template arguments.
  Out << printTemplateSpecializationArgs(ND);

  return Out.str();
}

std::string printTemplateSpecializationArgs(const NamedDecl &ND) {
  std::string TemplateArgs;
  llvm::raw_string_ostream OS(TemplateArgs);
  PrintingPolicy Policy(ND.getASTContext().getLangOpts());
  if (llvm::Optional<llvm::ArrayRef<TemplateArgumentLoc>> Args =
          getTemplateSpecializationArgLocs(ND)) {
    printTemplateArgumentList(OS, *Args, Policy);
  } else if (auto *Cls = llvm::dyn_cast<ClassTemplateSpecializationDecl>(&ND)) {
    if (const TypeSourceInfo *TSI = Cls->getTypeAsWritten()) {
      // ClassTemplateSpecializationDecls do not contain
      // TemplateArgumentTypeLocs, they only have TemplateArgumentTypes. So we
      // create a new argument location list from TypeSourceInfo.
      auto STL = TSI->getTypeLoc().getAs<TemplateSpecializationTypeLoc>();
      llvm::SmallVector<TemplateArgumentLoc> ArgLocs;
      ArgLocs.reserve(STL.getNumArgs());
      for (unsigned I = 0; I < STL.getNumArgs(); ++I)
        ArgLocs.push_back(STL.getArgLoc(I));
      printTemplateArgumentList(OS, ArgLocs, Policy);
    } else {
      // FIXME: Fix cases when getTypeAsWritten returns null inside clang AST,
      // e.g. friend decls. Currently we fallback to Template Arguments without
      // location information.
      printTemplateArgumentList(OS, Cls->getTemplateArgs().asArray(), Policy);
    }
  }
  OS.flush();
  return TemplateArgs;
}

std::string printNamespaceScope(const DeclContext &DC) {
  for (const auto *Ctx = &DC; Ctx != nullptr; Ctx = Ctx->getParent())
    if (const auto *NS = dyn_cast<NamespaceDecl>(Ctx))
      if (!NS->isAnonymousNamespace() && !NS->isInlineNamespace())
        return printQualifiedName(*NS) + "::";
  return "";
}

static llvm::StringRef
getNameOrErrForObjCInterface(const ObjCInterfaceDecl *ID) {
  return ID ? ID->getName() : "<<error-type>>";
}

std::string printObjCMethod(const ObjCMethodDecl &Method) {
  std::string Name;
  llvm::raw_string_ostream OS(Name);

  OS << (Method.isInstanceMethod() ? '-' : '+') << '[';

  // Should always be true.
  if (const ObjCContainerDecl *C =
          dyn_cast<ObjCContainerDecl>(Method.getDeclContext()))
    OS << printObjCContainer(*C);

  Method.getSelector().print(OS << ' ');
  if (Method.isVariadic())
    OS << ", ...";

  OS << ']';
  OS.flush();
  return Name;
}

std::string printObjCContainer(const ObjCContainerDecl &C) {
  if (const ObjCCategoryDecl *Category = dyn_cast<ObjCCategoryDecl>(&C)) {
    std::string Name;
    llvm::raw_string_ostream OS(Name);
    const ObjCInterfaceDecl *Class = Category->getClassInterface();
    OS << getNameOrErrForObjCInterface(Class) << '(' << Category->getName()
       << ')';
    OS.flush();
    return Name;
  }
  if (const ObjCCategoryImplDecl *CID = dyn_cast<ObjCCategoryImplDecl>(&C)) {
    std::string Name;
    llvm::raw_string_ostream OS(Name);
    const ObjCInterfaceDecl *Class = CID->getClassInterface();
    OS << getNameOrErrForObjCInterface(Class) << '(' << CID->getName() << ')';
    OS.flush();
    return Name;
  }
  return C.getNameAsString();
}

SymbolID getSymbolID(const Decl *D) {
  llvm::SmallString<128> USR;
  if (index::generateUSRForDecl(D, USR))
    return {};
  return SymbolID(USR);
}

SymbolID getSymbolID(const llvm::StringRef MacroName, const MacroInfo *MI,
                     const SourceManager &SM) {
  if (MI == nullptr)
    return {};
  llvm::SmallString<128> USR;
  if (index::generateUSRForMacro(MacroName, MI->getDefinitionLoc(), SM, USR))
    return {};
  return SymbolID(USR);
}

std::string printType(const QualType QT, const DeclContext &CurContext) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  PrintingPolicy PP(CurContext.getParentASTContext().getPrintingPolicy());
  PP.SuppressTagKeyword = true;
  PP.SuppressUnwrittenScope = true;

  class PrintCB : public PrintingCallbacks {
  public:
    PrintCB(const DeclContext *CurContext) : CurContext(CurContext) {}
    virtual ~PrintCB() {}
    virtual bool isScopeVisible(const DeclContext *DC) const override {
      return DC->Encloses(CurContext);
    }

  private:
    const DeclContext *CurContext;
  };
  PrintCB PCB(&CurContext);
  PP.Callbacks = &PCB;

  QT.print(OS, PP);
  return OS.str();
}

bool hasReservedName(const Decl &D) {
  if (const auto *ND = llvm::dyn_cast<NamedDecl>(&D))
    if (const auto *II = ND->getIdentifier())
      return isReservedName(II->getName());
  return false;
}

bool hasReservedScope(const DeclContext &DC) {
  for (const DeclContext *D = &DC; D; D = D->getParent()) {
    if (D->isTransparentContext() || D->isInlineNamespace())
      continue;
    if (const auto *ND = llvm::dyn_cast<NamedDecl>(D))
      if (hasReservedName(*ND))
        return true;
  }
  return false;
}

QualType declaredType(const TypeDecl *D) {
  if (const auto *CTSD = llvm::dyn_cast<ClassTemplateSpecializationDecl>(D))
    if (const auto *TSI = CTSD->getTypeAsWritten())
      return TSI->getType();
  return D->getASTContext().getTypeDeclType(D);
}

namespace {
/// Computes the deduced type at a given location by visiting the relevant
/// nodes. We use this to display the actual type when hovering over an "auto"
/// keyword or "decltype()" expression.
/// FIXME: This could have been a lot simpler by visiting AutoTypeLocs but it
/// seems that the AutoTypeLocs that can be visited along with their AutoType do
/// not have the deduced type set. Instead, we have to go to the appropriate
/// DeclaratorDecl/FunctionDecl and work our back to the AutoType that does have
/// a deduced type set. The AST should be improved to simplify this scenario.
class DeducedTypeVisitor : public RecursiveASTVisitor<DeducedTypeVisitor> {
  SourceLocation SearchedLocation;

public:
  DeducedTypeVisitor(SourceLocation SearchedLocation)
      : SearchedLocation(SearchedLocation) {}

  // Handle auto initializers:
  //- auto i = 1;
  //- decltype(auto) i = 1;
  //- auto& i = 1;
  //- auto* i = &a;
  bool VisitDeclaratorDecl(DeclaratorDecl *D) {
    if (!D->getTypeSourceInfo() ||
        D->getTypeSourceInfo()->getTypeLoc().getBeginLoc() != SearchedLocation)
      return true;

    if (auto *AT = D->getType()->getContainedAutoType()) {
      DeducedType = AT->desugar();
    }
    return true;
  }

  // Handle auto return types:
  //- auto foo() {}
  //- auto& foo() {}
  //- auto foo() -> int {}
  //- auto foo() -> decltype(1+1) {}
  //- operator auto() const { return 10; }
  bool VisitFunctionDecl(FunctionDecl *D) {
    if (!D->getTypeSourceInfo())
      return true;
    // Loc of auto in return type (c++14).
    auto CurLoc = D->getReturnTypeSourceRange().getBegin();
    // Loc of "auto" in operator auto()
    if (CurLoc.isInvalid() && isa<CXXConversionDecl>(D))
      CurLoc = D->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
    // Loc of "auto" in function with trailing return type (c++11).
    if (CurLoc.isInvalid())
      CurLoc = D->getSourceRange().getBegin();
    if (CurLoc != SearchedLocation)
      return true;

    const AutoType *AT = D->getReturnType()->getContainedAutoType();
    if (AT && !AT->getDeducedType().isNull()) {
      DeducedType = AT->getDeducedType();
    } else if (auto *DT = dyn_cast<DecltypeType>(D->getReturnType())) {
      // auto in a trailing return type just points to a DecltypeType and
      // getContainedAutoType does not unwrap it.
      if (!DT->getUnderlyingType().isNull())
        DeducedType = DT->getUnderlyingType();
    } else if (!D->getReturnType().isNull()) {
      DeducedType = D->getReturnType();
    }
    return true;
  }

  // Handle non-auto decltype, e.g.:
  // - auto foo() -> decltype(expr) {}
  // - decltype(expr);
  bool VisitDecltypeTypeLoc(DecltypeTypeLoc TL) {
    if (TL.getBeginLoc() != SearchedLocation)
      return true;

    // A DecltypeType's underlying type can be another DecltypeType! E.g.
    //  int I = 0;
    //  decltype(I) J = I;
    //  decltype(J) K = J;
    const DecltypeType *DT = dyn_cast<DecltypeType>(TL.getTypePtr());
    while (DT && !DT->getUnderlyingType().isNull()) {
      DeducedType = DT->getUnderlyingType();
      DT = dyn_cast<DecltypeType>(DeducedType.getTypePtr());
    }
    return true;
  }

  QualType DeducedType;
};
} // namespace

llvm::Optional<QualType> getDeducedType(ASTContext &ASTCtx,
                                        SourceLocation Loc) {
  if (!Loc.isValid())
    return {};
  DeducedTypeVisitor V(Loc);
  V.TraverseAST(ASTCtx);
  if (V.DeducedType.isNull())
    return llvm::None;
  return V.DeducedType;
}

std::vector<const Attr *> getAttributes(const DynTypedNode &N) {
  std::vector<const Attr *> Result;
  if (const auto *TL = N.get<TypeLoc>()) {
    for (AttributedTypeLoc ATL = TL->getAs<AttributedTypeLoc>(); !ATL.isNull();
         ATL = ATL.getModifiedLoc().getAs<AttributedTypeLoc>()) {
      if (const Attr *A = ATL.getAttr())
        Result.push_back(A);
      assert(!ATL.getModifiedLoc().isNull());
    }
  }
  if (const auto *S = N.get<AttributedStmt>()) {
    for (; S != nullptr; S = dyn_cast<AttributedStmt>(S->getSubStmt()))
      for (const Attr *A : S->getAttrs())
        if (A)
          Result.push_back(A);
  }
  if (const auto *D = N.get<Decl>()) {
    for (const Attr *A : D->attrs())
      if (A)
        Result.push_back(A);
  }
  return Result;
}

std::string getQualification(ASTContext &Context,
                             const DeclContext *DestContext,
                             SourceLocation InsertionPoint,
                             const NamedDecl *ND) {
  auto VisibleNamespaceDecls =
      getUsingNamespaceDirectives(DestContext, InsertionPoint);
  return getQualification(
      Context, DestContext, ND->getDeclContext(),
      [&](NestedNameSpecifier *NNS) {
        if (NNS->getKind() != NestedNameSpecifier::Namespace)
          return false;
        const auto *CanonNSD = NNS->getAsNamespace()->getCanonicalDecl();
        return llvm::any_of(VisibleNamespaceDecls,
                            [CanonNSD](const NamespaceDecl *NSD) {
                              return NSD->getCanonicalDecl() == CanonNSD;
                            });
      });
}

std::string getQualification(ASTContext &Context,
                             const DeclContext *DestContext,
                             const NamedDecl *ND,
                             llvm::ArrayRef<std::string> VisibleNamespaces) {
  for (llvm::StringRef NS : VisibleNamespaces) {
    assert(NS.endswith("::"));
    (void)NS;
  }
  return getQualification(
      Context, DestContext, ND->getDeclContext(),
      [&](NestedNameSpecifier *NNS) {
        return llvm::any_of(VisibleNamespaces, [&](llvm::StringRef Namespace) {
          std::string NS;
          llvm::raw_string_ostream OS(NS);
          NNS->print(OS, Context.getPrintingPolicy());
          return OS.str() == Namespace;
        });
      });
}

bool hasUnstableLinkage(const Decl *D) {
  // Linkage of a ValueDecl depends on the type.
  // If that's not deduced yet, deducing it may change the linkage.
  auto *VD = llvm::dyn_cast_or_null<ValueDecl>(D);
  return VD && !VD->getType().isNull() && VD->getType()->isUndeducedType();
}

bool isDeeplyNested(const Decl *D, unsigned MaxDepth) {
  size_t ContextDepth = 0;
  for (auto *Ctx = D->getDeclContext(); Ctx && !Ctx->isTranslationUnit();
       Ctx = Ctx->getParent()) {
    if (++ContextDepth == MaxDepth)
      return true;
  }
  return false;
}
} // namespace clangd
} // namespace clang
