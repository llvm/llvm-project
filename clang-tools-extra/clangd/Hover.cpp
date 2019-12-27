//===--- Hover.cpp - Information about code at the cursor location --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Hover.h"

#include "AST.h"
#include "CodeCompletionStrings.h"
#include "FindTarget.h"
#include "FormattedString.h"
#include "Logger.h"
#include "Selection.h"
#include "SourceCode.h"
#include "index/SymbolCollector.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {
namespace {

PrintingPolicy printingPolicyForDecls(PrintingPolicy Base) {
  PrintingPolicy Policy(Base);

  Policy.AnonymousTagLocations = false;
  Policy.TerseOutput = true;
  Policy.PolishForDeclaration = true;
  Policy.ConstantsAsWritten = true;
  Policy.SuppressTagKeyword = false;

  return Policy;
}

/// Given a declaration \p D, return a human-readable string representing the
/// local scope in which it is declared, i.e. class(es) and method name. Returns
/// an empty string if it is not local.
std::string getLocalScope(const Decl *D) {
  std::vector<std::string> Scopes;
  const DeclContext *DC = D->getDeclContext();
  auto GetName = [](const TypeDecl *D) {
    if (!D->getDeclName().isEmpty()) {
      PrintingPolicy Policy = D->getASTContext().getPrintingPolicy();
      Policy.SuppressScope = true;
      return declaredType(D).getAsString(Policy);
    }
    if (auto RD = dyn_cast<RecordDecl>(D))
      return ("(anonymous " + RD->getKindName() + ")").str();
    return std::string("");
  };
  while (DC) {
    if (const TypeDecl *TD = dyn_cast<TypeDecl>(DC))
      Scopes.push_back(GetName(TD));
    else if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(DC))
      Scopes.push_back(FD->getNameAsString());
    DC = DC->getParent();
  }

  return llvm::join(llvm::reverse(Scopes), "::");
}

/// Returns the human-readable representation for namespace containing the
/// declaration \p D. Returns empty if it is contained global namespace.
std::string getNamespaceScope(const Decl *D) {
  const DeclContext *DC = D->getDeclContext();

  if (const TagDecl *TD = dyn_cast<TagDecl>(DC))
    return getNamespaceScope(TD);
  if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(DC))
    return getNamespaceScope(FD);
  if (const NamespaceDecl *NSD = dyn_cast<NamespaceDecl>(DC)) {
    // Skip inline/anon namespaces.
    if (NSD->isInline() || NSD->isAnonymousNamespace())
      return getNamespaceScope(NSD);
  }
  if (const NamedDecl *ND = dyn_cast<NamedDecl>(DC))
    return printQualifiedName(*ND);

  return "";
}

std::string printDefinition(const Decl *D) {
  std::string Definition;
  llvm::raw_string_ostream OS(Definition);
  PrintingPolicy Policy =
      printingPolicyForDecls(D->getASTContext().getPrintingPolicy());
  Policy.IncludeTagDefinition = false;
  Policy.SuppressTemplateArgsInCXXConstructors = true;
  D->print(OS, Policy);
  OS.flush();
  return Definition;
}

void printParams(llvm::raw_ostream &OS,
                 const std::vector<HoverInfo::Param> &Params) {
  for (size_t I = 0, E = Params.size(); I != E; ++I) {
    if (I)
      OS << ", ";
    OS << Params.at(I);
  }
}

std::vector<HoverInfo::Param>
fetchTemplateParameters(const TemplateParameterList *Params,
                        const PrintingPolicy &PP) {
  assert(Params);
  std::vector<HoverInfo::Param> TempParameters;

  for (const Decl *Param : *Params) {
    HoverInfo::Param P;
    P.Type.emplace();
    if (const auto TTP = dyn_cast<TemplateTypeParmDecl>(Param)) {
      P.Type = TTP->wasDeclaredWithTypename() ? "typename" : "class";
      if (TTP->isParameterPack())
        *P.Type += "...";

      if (!TTP->getName().empty())
        P.Name = TTP->getNameAsString();
      if (TTP->hasDefaultArgument())
        P.Default = TTP->getDefaultArgument().getAsString(PP);
    } else if (const auto NTTP = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
      if (IdentifierInfo *II = NTTP->getIdentifier())
        P.Name = II->getName().str();

      llvm::raw_string_ostream Out(*P.Type);
      NTTP->getType().print(Out, PP);
      if (NTTP->isParameterPack())
        Out << "...";

      if (NTTP->hasDefaultArgument()) {
        P.Default.emplace();
        llvm::raw_string_ostream Out(*P.Default);
        NTTP->getDefaultArgument()->printPretty(Out, nullptr, PP);
      }
    } else if (const auto TTPD = dyn_cast<TemplateTemplateParmDecl>(Param)) {
      llvm::raw_string_ostream OS(*P.Type);
      OS << "template <";
      printParams(OS,
                  fetchTemplateParameters(TTPD->getTemplateParameters(), PP));
      OS << "> class"; // FIXME: TemplateTemplateParameter doesn't store the
                       // info on whether this param was a "typename" or
                       // "class".
      if (!TTPD->getName().empty())
        P.Name = TTPD->getNameAsString();
      if (TTPD->hasDefaultArgument()) {
        P.Default.emplace();
        llvm::raw_string_ostream Out(*P.Default);
        TTPD->getDefaultArgument().getArgument().print(PP, Out);
      }
    }
    TempParameters.push_back(std::move(P));
  }

  return TempParameters;
}

const FunctionDecl *getUnderlyingFunction(const Decl *D) {
  // Extract lambda from variables.
  if (const VarDecl *VD = llvm::dyn_cast<VarDecl>(D)) {
    auto QT = VD->getType();
    if (!QT.isNull()) {
      while (!QT->getPointeeType().isNull())
        QT = QT->getPointeeType();

      if (const auto *CD = QT->getAsCXXRecordDecl())
        return CD->getLambdaCallOperator();
    }
  }

  // Non-lambda functions.
  return D->getAsFunction();
}

// Returns the decl that should be used for querying comments, either from index
// or AST.
const NamedDecl *getDeclForComment(const NamedDecl *D) {
  if (auto *CTSD = llvm::dyn_cast<ClassTemplateSpecializationDecl>(D))
    if (!CTSD->isExplicitInstantiationOrSpecialization())
      return CTSD->getTemplateInstantiationPattern();
  if (auto *VTSD = llvm::dyn_cast<VarTemplateSpecializationDecl>(D))
    if (!VTSD->isExplicitInstantiationOrSpecialization())
      return VTSD->getTemplateInstantiationPattern();
  if (auto *FD = D->getAsFunction())
    if (FD->isTemplateInstantiation())
      return FD->getTemplateInstantiationPattern();
  return D;
}

// Look up information about D from the index, and add it to Hover.
void enhanceFromIndex(HoverInfo &Hover, const NamedDecl &ND,
                      const SymbolIndex *Index) {
  assert(&ND == getDeclForComment(&ND));
  // We only add documentation, so don't bother if we already have some.
  if (!Hover.Documentation.empty() || !Index)
    return;

  // Skip querying for non-indexable symbols, there's no point.
  // We're searching for symbols that might be indexed outside this main file.
  if (!SymbolCollector::shouldCollectSymbol(ND, ND.getASTContext(),
                                            SymbolCollector::Options(),
                                            /*IsMainFileOnly=*/false))
    return;
  auto ID = getSymbolID(&ND);
  if (!ID)
    return;
  LookupRequest Req;
  Req.IDs.insert(*ID);
  Index->lookup(
      Req, [&](const Symbol &S) { Hover.Documentation = S.Documentation; });
}

// Populates Type, ReturnType, and Parameters for function-like decls.
void fillFunctionTypeAndParams(HoverInfo &HI, const Decl *D,
                                      const FunctionDecl *FD,
                                      const PrintingPolicy &Policy) {
  HI.Parameters.emplace();
  for (const ParmVarDecl *PVD : FD->parameters()) {
    HI.Parameters->emplace_back();
    auto &P = HI.Parameters->back();
    if (!PVD->getType().isNull()) {
      P.Type.emplace();
      llvm::raw_string_ostream OS(*P.Type);
      PVD->getType().print(OS, Policy);
    } else {
      std::string Param;
      llvm::raw_string_ostream OS(Param);
      PVD->dump(OS);
      OS.flush();
      elog("Got param with null type: {0}", Param);
    }
    if (!PVD->getName().empty())
      P.Name = PVD->getNameAsString();
    if (PVD->hasDefaultArg()) {
      P.Default.emplace();
      llvm::raw_string_ostream Out(*P.Default);
      PVD->getDefaultArg()->printPretty(Out, nullptr, Policy);
    }
  }

  if (const auto* CCD = llvm::dyn_cast<CXXConstructorDecl>(FD)) {
    // Constructor's "return type" is the class type.
    HI.ReturnType = declaredType(CCD->getParent()).getAsString(Policy);
    // Don't provide any type for the constructor itself.
  } else if (llvm::isa<CXXDestructorDecl>(FD)){
    HI.ReturnType = "void";
  } else {
    HI.ReturnType = FD->getReturnType().getAsString(Policy);

    QualType FunctionType = FD->getType();
    if (const VarDecl *VD = llvm::dyn_cast<VarDecl>(D)) // Lambdas
      FunctionType = VD->getType().getDesugaredType(D->getASTContext());
    HI.Type = FunctionType.getAsString(Policy);
  }
  // FIXME: handle variadics.
}

llvm::Optional<std::string> printExprValue(const Expr *E,
                                           const ASTContext &Ctx) {
  Expr::EvalResult Constant;
  // Evaluating [[foo]]() as "&foo" isn't useful, and prevents us walking up
  // to the enclosing call.
  QualType T = E->getType();
  if (T.isNull() || T->isFunctionType() || T->isFunctionPointerType() ||
      T->isFunctionReferenceType())
    return llvm::None;
  // Attempt to evaluate. If expr is dependent, evaluation crashes!
  if (E->isValueDependent() || !E->EvaluateAsRValue(Constant, Ctx))
    return llvm::None;

  // Show enums symbolically, not numerically like APValue::printPretty().
  if (T->isEnumeralType() && Constant.Val.getInt().getMinSignedBits() <= 64) {
    // Compare to int64_t to avoid bit-width match requirements.
    int64_t Val = Constant.Val.getInt().getExtValue();
    for (const EnumConstantDecl *ECD :
         T->castAs<EnumType>()->getDecl()->enumerators())
      if (ECD->getInitVal() == Val)
        return llvm::formatv("{0} ({1})", ECD->getNameAsString(), Val).str();
  }
  return Constant.Val.getAsString(Ctx, E->getType());
}

llvm::Optional<std::string> printExprValue(const SelectionTree::Node *N,
                                           const ASTContext &Ctx) {
  for (; N; N = N->Parent) {
    // Try to evaluate the first evaluable enclosing expression.
    if (const Expr *E = N->ASTNode.get<Expr>()) {
      if (auto Val = printExprValue(E, Ctx))
        return Val;
    } else if (N->ASTNode.get<Decl>() || N->ASTNode.get<Stmt>()) {
      // Refuse to cross certain non-exprs. (TypeLoc are OK as part of Exprs).
      // This tries to ensure we're showing a value related to the cursor.
      break;
    }
  }
  return llvm::None;
}

/// Generate a \p Hover object given the declaration \p D.
HoverInfo getHoverContents(const Decl *D, const SymbolIndex *Index) {
  HoverInfo HI;
  const ASTContext &Ctx = D->getASTContext();

  HI.NamespaceScope = getNamespaceScope(D);
  if (!HI.NamespaceScope->empty())
    HI.NamespaceScope->append("::");
  HI.LocalScope = getLocalScope(D);
  if (!HI.LocalScope.empty())
    HI.LocalScope.append("::");

  PrintingPolicy Policy = printingPolicyForDecls(Ctx.getPrintingPolicy());
  if (const NamedDecl *ND = llvm::dyn_cast<NamedDecl>(D)) {
    HI.Name = printName(Ctx, *ND);
    ND = getDeclForComment(ND);
    HI.Documentation = getDeclComment(Ctx, *ND);
    enhanceFromIndex(HI, *ND, Index);
  }

  HI.Kind = index::getSymbolInfo(D).Kind;

  // Fill in template params.
  if (const TemplateDecl *TD = D->getDescribedTemplate()) {
    HI.TemplateParameters =
        fetchTemplateParameters(TD->getTemplateParameters(), Policy);
    D = TD;
  } else if (const FunctionDecl *FD = D->getAsFunction()) {
    if (const auto FTD = FD->getDescribedTemplate()) {
      HI.TemplateParameters =
          fetchTemplateParameters(FTD->getTemplateParameters(), Policy);
      D = FTD;
    }
  }

  // Fill in types and params.
  if (const FunctionDecl *FD = getUnderlyingFunction(D)) {
    fillFunctionTypeAndParams(HI, D, FD, Policy);
  } else if (const auto *VD = dyn_cast<ValueDecl>(D)) {
    HI.Type.emplace();
    llvm::raw_string_ostream OS(*HI.Type);
    VD->getType().print(OS, Policy);
  }

  // Fill in value with evaluated initializer if possible.
  if (const auto *Var = dyn_cast<VarDecl>(D)) {
    if (const Expr *Init = Var->getInit())
      HI.Value = printExprValue(Init, Ctx);
  } else if (const auto *ECD = dyn_cast<EnumConstantDecl>(D)) {
    // Dependent enums (e.g. nested in template classes) don't have values yet.
    if (!ECD->getType()->isDependentType())
      HI.Value = ECD->getInitVal().toString(10);
  }

  HI.Definition = printDefinition(D);
  return HI;
}

/// Generate a \p Hover object given the type \p T.
HoverInfo getHoverContents(QualType T, ASTContext &ASTCtx,
                           const SymbolIndex *Index) {
  HoverInfo HI;

  if (const auto *D = T->getAsTagDecl()) {
    HI.Name = printName(ASTCtx, *D);
    HI.Kind = index::getSymbolInfo(D).Kind;

    const auto *CommentD = getDeclForComment(D);
    HI.Documentation = getDeclComment(ASTCtx, *CommentD);
    enhanceFromIndex(HI, *CommentD, Index);
  } else {
    // Builtin types
    llvm::raw_string_ostream OS(HI.Name);
    PrintingPolicy Policy = printingPolicyForDecls(ASTCtx.getPrintingPolicy());
    T.print(OS, Policy);
  }
  return HI;
}

/// Generate a \p Hover object given the macro \p MacroDecl.
HoverInfo getHoverContents(const DefinedMacro &Macro, ParsedAST &AST) {
  HoverInfo HI;
  SourceManager &SM = AST.getSourceManager();
  HI.Name = Macro.Name;
  HI.Kind = index::SymbolKind::Macro;
  // FIXME: Populate documentation
  // FIXME: Pupulate parameters

  // Try to get the full definition, not just the name
  SourceLocation StartLoc = Macro.Info->getDefinitionLoc();
  SourceLocation EndLoc = Macro.Info->getDefinitionEndLoc();
  if (EndLoc.isValid()) {
    EndLoc = Lexer::getLocForEndOfToken(EndLoc, 0, SM, AST.getLangOpts());
    bool Invalid;
    StringRef Buffer = SM.getBufferData(SM.getFileID(StartLoc), &Invalid);
    if (!Invalid) {
      unsigned StartOffset = SM.getFileOffset(StartLoc);
      unsigned EndOffset = SM.getFileOffset(EndLoc);
      if (EndOffset <= Buffer.size() && StartOffset < EndOffset)
        HI.Definition =
            ("#define " + Buffer.substr(StartOffset, EndOffset - StartOffset))
                .str();
    }
  }
  return HI;
}
} // namespace

llvm::Optional<HoverInfo> getHover(ParsedAST &AST, Position Pos,
                                   format::FormatStyle Style,
                                   const SymbolIndex *Index) {
  const SourceManager &SM = AST.getSourceManager();
  llvm::Optional<HoverInfo> HI;
  SourceLocation SourceLocationBeg = SM.getMacroArgExpandedLocation(
      getBeginningOfIdentifier(Pos, SM, AST.getLangOpts()));

  if (auto Deduced = getDeducedType(AST.getASTContext(), SourceLocationBeg)) {
    HI = getHoverContents(*Deduced, AST.getASTContext(), Index);
  } else if (auto M = locateMacroAt(SourceLocationBeg, AST.getPreprocessor())) {
    HI = getHoverContents(*M, AST);
  } else {
    auto Offset = positionToOffset(SM.getBufferData(SM.getMainFileID()), Pos);
    if (!Offset) {
      llvm::consumeError(Offset.takeError());
      return llvm::None;
    }
    SelectionTree Selection(AST.getASTContext(), AST.getTokens(), *Offset);
    std::vector<const Decl *> Result;
    if (const SelectionTree::Node *N = Selection.commonAncestor()) {
      auto Decls = explicitReferenceTargets(N->ASTNode, DeclRelation::Alias);
      if (!Decls.empty()) {
        HI = getHoverContents(Decls.front(), Index);
        // Look for a close enclosing expression to show the value of.
        if (!HI->Value)
          HI->Value = printExprValue(N, AST.getASTContext());
      }
      // FIXME: support hovers for other nodes?
      //  - certain expressions (sizeof etc)
      //  - built-in types
      //  - literals (esp user-defined)
    }
  }

  if (!HI)
    return llvm::None;

  auto Replacements = format::reformat(
      Style, HI->Definition, tooling::Range(0, HI->Definition.size()));
  if (auto Formatted =
          tooling::applyAllReplacements(HI->Definition, Replacements))
    HI->Definition = *Formatted;

  HI->SymRange = getTokenRange(AST.getSourceManager(),
                               AST.getLangOpts(), SourceLocationBeg);
  return HI;
}

markup::Document HoverInfo::present() const {
  markup::Document Output;
  if (NamespaceScope) {
    auto &P = Output.addParagraph();
    P.appendText("Declared in");
    // Drop trailing "::".
    if (!LocalScope.empty())
      P.appendCode(llvm::StringRef(LocalScope).drop_back(2));
    else if (NamespaceScope->empty())
      P.appendCode("global namespace");
    else
      P.appendCode(llvm::StringRef(*NamespaceScope).drop_back(2));
  }

  if (!Definition.empty()) {
    Output.addCodeBlock(Definition);
  } else {
    // Builtin types
    Output.addCodeBlock(Name);
  }

  if (!Documentation.empty())
    Output.addParagraph().appendText(Documentation);
  return Output;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const HoverInfo::Param &P) {
  std::vector<llvm::StringRef> Output;
  if (P.Type)
    Output.push_back(*P.Type);
  if (P.Name)
    Output.push_back(*P.Name);
  OS << llvm::join(Output, " ");
  if (P.Default)
    OS << " = " << *P.Default;
  return OS;
}

} // namespace clangd
} // namespace clang
