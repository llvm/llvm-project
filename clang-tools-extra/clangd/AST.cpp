//===--- AST.cpp - Utility AST functions  -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"

#include "SourceCode.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Index/USRGeneration.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"

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

SourceLocation findName(const clang::Decl *D) {
  return D->getLocation();
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
    if (auto *Cls = llvm::dyn_cast<RecordDecl>(&ND))
      return ("(anonymous " + Cls->getKindName() + ")").str();
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
      llvm::SmallVector<TemplateArgumentLoc, 8> ArgLocs;
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

llvm::Optional<SymbolID> getSymbolID(const Decl *D) {
  llvm::SmallString<128> USR;
  if (index::generateUSRForDecl(D, USR))
    return None;
  return SymbolID(USR);
}

llvm::Optional<SymbolID> getSymbolID(const llvm::StringRef MacroName,
                                     const MacroInfo *MI,
                                     const SourceManager &SM) {
  if (MI == nullptr)
    return None;
  llvm::SmallString<128> USR;
  if (index::generateUSRForMacro(MacroName, MI->getDefinitionLoc(), SM, USR))
    return None;
  return SymbolID(USR);
}

std::string shortenNamespace(const llvm::StringRef OriginalName,
                             const llvm::StringRef CurrentNamespace) {
  llvm::SmallVector<llvm::StringRef, 8> OriginalParts;
  llvm::SmallVector<llvm::StringRef, 8> CurrentParts;
  llvm::SmallVector<llvm::StringRef, 8> Result;
  OriginalName.split(OriginalParts, "::");
  CurrentNamespace.split(CurrentParts, "::");
  auto MinLength = std::min(CurrentParts.size(), OriginalParts.size());

  unsigned DifferentAt = 0;
  while (DifferentAt < MinLength &&
      CurrentParts[DifferentAt] == OriginalParts[DifferentAt]) {
    DifferentAt++;
  }

  for (unsigned i = DifferentAt; i < OriginalParts.size(); ++i) {
    Result.push_back(OriginalParts[i]);
  }
  return join(Result, "::");
}

std::string printType(const QualType QT, const DeclContext & Context){
  PrintingPolicy PP(Context.getParentASTContext().getPrintingPolicy());
  PP.SuppressUnwrittenScope = 1;
  PP.SuppressTagKeyword = 1;
  return shortenNamespace(
      QT.getAsString(PP),
      printNamespaceScope(Context) );
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
      if (!AT->getDeducedType().isNull())
        DeducedType = AT->getDeducedType();
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
    if (CurLoc.isInvalid() && dyn_cast<CXXConversionDecl>(D))
      CurLoc = D->getTypeSourceInfo()->getTypeLoc().getBeginLoc();
    // Loc of "auto" in function with traling return type (c++11).
    if (CurLoc.isInvalid())
      CurLoc = D->getSourceRange().getBegin();
    if (CurLoc != SearchedLocation)
      return true;

    const AutoType *AT = D->getReturnType()->getContainedAutoType();
    if (AT && !AT->getDeducedType().isNull()) {
      DeducedType = AT->getDeducedType();
    } else if (auto DT = dyn_cast<DecltypeType>(D->getReturnType())) {
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
  Token Tok;
  // Only try to find a deduced type if the token is auto or decltype.
  if (!Loc.isValid() ||
      Lexer::getRawToken(Loc, Tok, ASTCtx.getSourceManager(),
                         ASTCtx.getLangOpts(), false) ||
      !Tok.is(tok::raw_identifier) ||
      !(Tok.getRawIdentifier() == "auto" ||
        Tok.getRawIdentifier() == "decltype")) {
    return {};
  }
  DeducedTypeVisitor V(Loc);
  V.TraverseAST(ASTCtx);
  if (V.DeducedType.isNull())
    return llvm::None;
  return V.DeducedType;
}

} // namespace clangd
} // namespace clang
