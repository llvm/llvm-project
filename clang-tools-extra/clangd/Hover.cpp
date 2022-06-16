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
#include "Config.h"
#include "FindTarget.h"
#include "ParsedAST.h"
#include "Selection.h"
#include "SourceCode.h"
#include "index/SymbolCollector.h"
#include "support/Markup.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Index/IndexSymbol.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace clang {
namespace clangd {
namespace {

PrintingPolicy getPrintingPolicy(PrintingPolicy Base) {
  Base.AnonymousTagLocations = false;
  Base.TerseOutput = true;
  Base.PolishForDeclaration = true;
  Base.ConstantsAsWritten = true;
  Base.SuppressTemplateArgsInCXXConstructors = true;
  return Base;
}

/// Given a declaration \p D, return a human-readable string representing the
/// local scope in which it is declared, i.e. class(es) and method name. Returns
/// an empty string if it is not local.
std::string getLocalScope(const Decl *D) {
  std::vector<std::string> Scopes;
  const DeclContext *DC = D->getDeclContext();

  // ObjC scopes won't have multiple components for us to join, instead:
  // - Methods: "-[Class methodParam1:methodParam2]"
  // - Classes, categories, and protocols: "MyClass(Category)"
  if (const ObjCMethodDecl *MD = dyn_cast<ObjCMethodDecl>(DC))
    return printObjCMethod(*MD);
  if (const ObjCContainerDecl *CD = dyn_cast<ObjCContainerDecl>(DC))
    return printObjCContainer(*CD);

  auto GetName = [](const TypeDecl *D) {
    if (!D->getDeclName().isEmpty()) {
      PrintingPolicy Policy = D->getASTContext().getPrintingPolicy();
      Policy.SuppressScope = true;
      return declaredType(D).getAsString(Policy);
    }
    if (auto *RD = dyn_cast<RecordDecl>(D))
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

  // ObjC does not have the concept of namespaces, so instead we support
  // local scopes.
  if (isa<ObjCMethodDecl, ObjCContainerDecl>(DC))
    return "";

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

std::string printDefinition(const Decl *D, PrintingPolicy PP,
                            const syntax::TokenBuffer &TB) {
  if (auto *VD = llvm::dyn_cast<VarDecl>(D)) {
    if (auto *IE = VD->getInit()) {
      // Initializers might be huge and result in lots of memory allocations in
      // some catostrophic cases. Such long lists are not useful in hover cards
      // anyway.
      if (200 < TB.expandedTokens(IE->getSourceRange()).size())
        PP.SuppressInitializers = true;
    }
  }
  std::string Definition;
  llvm::raw_string_ostream OS(Definition);
  D->print(OS, PP);
  OS.flush();
  return Definition;
}

const char *getMarkdownLanguage(const ASTContext &Ctx) {
  const auto &LangOpts = Ctx.getLangOpts();
  if (LangOpts.ObjC && LangOpts.CPlusPlus)
    return "objective-cpp";
  return LangOpts.ObjC ? "objective-c" : "cpp";
}

HoverInfo::PrintedType printType(QualType QT, ASTContext &ASTCtx,
                                 const PrintingPolicy &PP) {
  // TypePrinter doesn't resolve decltypes, so resolve them here.
  // FIXME: This doesn't handle composite types that contain a decltype in them.
  // We should rather have a printing policy for that.
  while (!QT.isNull() && QT->isDecltypeType())
    QT = QT->castAs<DecltypeType>()->getUnderlyingType();
  HoverInfo::PrintedType Result;
  llvm::raw_string_ostream OS(Result.Type);
  // Special case: if the outer type is a tag type without qualifiers, then
  // include the tag for extra clarity.
  // This isn't very idiomatic, so don't attempt it for complex cases, including
  // pointers/references, template specializations, etc.
  if (!QT.isNull() && !QT.hasQualifiers() && PP.SuppressTagKeyword) {
    if (auto *TT = llvm::dyn_cast<TagType>(QT.getTypePtr()))
      OS << TT->getDecl()->getKindName() << " ";
  }
  QT.print(OS, PP);
  OS.flush();

  const Config &Cfg = Config::current();
  if (!QT.isNull() && Cfg.Hover.ShowAKA) {
    bool ShouldAKA = false;
    QualType DesugaredTy = clang::desugarForDiagnostic(ASTCtx, QT, ShouldAKA);
    if (ShouldAKA)
      Result.AKA = DesugaredTy.getAsString(PP);
  }
  return Result;
}

HoverInfo::PrintedType printType(const TemplateTypeParmDecl *TTP) {
  HoverInfo::PrintedType Result;
  Result.Type = TTP->wasDeclaredWithTypename() ? "typename" : "class";
  if (TTP->isParameterPack())
    Result.Type += "...";
  return Result;
}

HoverInfo::PrintedType printType(const NonTypeTemplateParmDecl *NTTP,
                                 const PrintingPolicy &PP) {
  auto PrintedType = printType(NTTP->getType(), NTTP->getASTContext(), PP);
  if (NTTP->isParameterPack()) {
    PrintedType.Type += "...";
    if (PrintedType.AKA)
      *PrintedType.AKA += "...";
  }
  return PrintedType;
}

HoverInfo::PrintedType printType(const TemplateTemplateParmDecl *TTP,
                                 const PrintingPolicy &PP) {
  HoverInfo::PrintedType Result;
  llvm::raw_string_ostream OS(Result.Type);
  OS << "template <";
  llvm::StringRef Sep = "";
  for (const Decl *Param : *TTP->getTemplateParameters()) {
    OS << Sep;
    Sep = ", ";
    if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(Param))
      OS << printType(TTP).Type;
    else if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Param))
      OS << printType(NTTP, PP).Type;
    else if (const auto *TTPD = dyn_cast<TemplateTemplateParmDecl>(Param))
      OS << printType(TTPD, PP).Type;
  }
  // FIXME: TemplateTemplateParameter doesn't store the info on whether this
  // param was a "typename" or "class".
  OS << "> class";
  OS.flush();
  return Result;
}

std::vector<HoverInfo::Param>
fetchTemplateParameters(const TemplateParameterList *Params,
                        const PrintingPolicy &PP) {
  assert(Params);
  std::vector<HoverInfo::Param> TempParameters;

  for (const Decl *Param : *Params) {
    HoverInfo::Param P;
    if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(Param)) {
      P.Type = printType(TTP);

      if (!TTP->getName().empty())
        P.Name = TTP->getNameAsString();

      if (TTP->hasDefaultArgument())
        P.Default = TTP->getDefaultArgument().getAsString(PP);
    } else if (const auto *NTTP = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
      P.Type = printType(NTTP, PP);

      if (IdentifierInfo *II = NTTP->getIdentifier())
        P.Name = II->getName().str();

      if (NTTP->hasDefaultArgument()) {
        P.Default.emplace();
        llvm::raw_string_ostream Out(*P.Default);
        NTTP->getDefaultArgument()->printPretty(Out, nullptr, PP);
      }
    } else if (const auto *TTPD = dyn_cast<TemplateTemplateParmDecl>(Param)) {
      P.Type = printType(TTPD, PP);

      if (!TTPD->getName().empty())
        P.Name = TTPD->getNameAsString();

      if (TTPD->hasDefaultArgument()) {
        P.Default.emplace();
        llvm::raw_string_ostream Out(*P.Default);
        TTPD->getDefaultArgument().getArgument().print(PP, Out,
                                                       /*IncludeType*/ false);
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
  const NamedDecl *DeclForComment = D;
  if (const auto *TSD = llvm::dyn_cast<ClassTemplateSpecializationDecl>(D)) {
    // Template may not be instantiated e.g. if the type didn't need to be
    // complete; fallback to primary template.
    if (TSD->getTemplateSpecializationKind() == TSK_Undeclared)
      DeclForComment = TSD->getSpecializedTemplate();
    else if (const auto *TIP = TSD->getTemplateInstantiationPattern())
      DeclForComment = TIP;
  } else if (const auto *TSD =
                 llvm::dyn_cast<VarTemplateSpecializationDecl>(D)) {
    if (TSD->getTemplateSpecializationKind() == TSK_Undeclared)
      DeclForComment = TSD->getSpecializedTemplate();
    else if (const auto *TIP = TSD->getTemplateInstantiationPattern())
      DeclForComment = TIP;
  } else if (const auto *FD = D->getAsFunction())
    if (const auto *TIP = FD->getTemplateInstantiationPattern())
      DeclForComment = TIP;
  // Ensure that getDeclForComment(getDeclForComment(X)) = getDeclForComment(X).
  // This is usually not needed, but in strange cases of comparision operators
  // being instantiated from spasceship operater, which itself is a template
  // instantiation the recursrive call is necessary.
  if (D != DeclForComment)
    DeclForComment = getDeclForComment(DeclForComment);
  return DeclForComment;
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
  Req.IDs.insert(ID);
  Index->lookup(Req, [&](const Symbol &S) {
    Hover.Documentation = std::string(S.Documentation);
  });
}

// Default argument might exist but be unavailable, in the case of unparsed
// arguments for example. This function returns the default argument if it is
// available.
const Expr *getDefaultArg(const ParmVarDecl *PVD) {
  // Default argument can be unparsed or uninstantiated. For the former we
  // can't do much, as token information is only stored in Sema and not
  // attached to the AST node. For the latter though, it is safe to proceed as
  // the expression is still valid.
  if (!PVD->hasDefaultArg() || PVD->hasUnparsedDefaultArg())
    return nullptr;
  return PVD->hasUninstantiatedDefaultArg() ? PVD->getUninstantiatedDefaultArg()
                                            : PVD->getDefaultArg();
}

HoverInfo::Param toHoverInfoParam(const ParmVarDecl *PVD,
                                  const PrintingPolicy &PP) {
  HoverInfo::Param Out;
  Out.Type = printType(PVD->getType(), PVD->getASTContext(), PP);
  if (!PVD->getName().empty())
    Out.Name = PVD->getNameAsString();
  if (const Expr *DefArg = getDefaultArg(PVD)) {
    Out.Default.emplace();
    llvm::raw_string_ostream OS(*Out.Default);
    DefArg->printPretty(OS, nullptr, PP);
  }
  return Out;
}

// Populates Type, ReturnType, and Parameters for function-like decls.
void fillFunctionTypeAndParams(HoverInfo &HI, const Decl *D,
                               const FunctionDecl *FD,
                               const PrintingPolicy &PP) {
  HI.Parameters.emplace();
  for (const ParmVarDecl *PVD : FD->parameters())
    HI.Parameters->emplace_back(toHoverInfoParam(PVD, PP));

  // We don't want any type info, if name already contains it. This is true for
  // constructors/destructors and conversion operators.
  const auto NK = FD->getDeclName().getNameKind();
  if (NK == DeclarationName::CXXConstructorName ||
      NK == DeclarationName::CXXDestructorName ||
      NK == DeclarationName::CXXConversionFunctionName)
    return;

  HI.ReturnType = printType(FD->getReturnType(), FD->getASTContext(), PP);
  QualType QT = FD->getType();
  if (const VarDecl *VD = llvm::dyn_cast<VarDecl>(D)) // Lambdas
    QT = VD->getType().getDesugaredType(D->getASTContext());
  HI.Type = printType(QT, D->getASTContext(), PP);
  // FIXME: handle variadics.
}

// Non-negative numbers are printed using min digits
// 0     => 0x0
// 100   => 0x64
// Negative numbers are sign-extended to 32/64 bits
// -2    => 0xfffffffe
// -2^32 => 0xfffffffeffffffff
static llvm::FormattedNumber printHex(const llvm::APSInt &V) {
  uint64_t Bits = V.getExtValue();
  if (V.isNegative() && V.getMinSignedBits() <= 32)
    return llvm::format_hex(uint32_t(Bits), 0);
  return llvm::format_hex(Bits, 0);
}

llvm::Optional<std::string> printExprValue(const Expr *E,
                                           const ASTContext &Ctx) {
  // InitListExpr has two forms, syntactic and semantic. They are the same thing
  // (refer to a same AST node) in most cases.
  // When they are different, RAV returns the syntactic form, and we should feed
  // the semantic form to EvaluateAsRValue.
  if (const auto *ILE = llvm::dyn_cast<InitListExpr>(E)) {
    if (!ILE->isSemanticForm())
      E = ILE->getSemanticForm();
  }

  // Evaluating [[foo]]() as "&foo" isn't useful, and prevents us walking up
  // to the enclosing call. Evaluating an expression of void type doesn't
  // produce a meaningful result.
  QualType T = E->getType();
  if (T.isNull() || T->isFunctionType() || T->isFunctionPointerType() ||
      T->isFunctionReferenceType() || T->isVoidType())
    return llvm::None;

  Expr::EvalResult Constant;
  // Attempt to evaluate. If expr is dependent, evaluation crashes!
  if (E->isValueDependent() || !E->EvaluateAsRValue(Constant, Ctx) ||
      // Disable printing for record-types, as they are usually confusing and
      // might make clang crash while printing the expressions.
      Constant.Val.isStruct() || Constant.Val.isUnion())
    return llvm::None;

  // Show enums symbolically, not numerically like APValue::printPretty().
  if (T->isEnumeralType() && Constant.Val.getInt().getMinSignedBits() <= 64) {
    // Compare to int64_t to avoid bit-width match requirements.
    int64_t Val = Constant.Val.getInt().getExtValue();
    for (const EnumConstantDecl *ECD :
         T->castAs<EnumType>()->getDecl()->enumerators())
      if (ECD->getInitVal() == Val)
        return llvm::formatv("{0} ({1})", ECD->getNameAsString(),
                             printHex(Constant.Val.getInt()))
            .str();
  }
  // Show hex value of integers if they're at least 10 (or negative!)
  if (T->isIntegralOrEnumerationType() &&
      Constant.Val.getInt().getMinSignedBits() <= 64 &&
      Constant.Val.getInt().uge(10))
    return llvm::formatv("{0} ({1})", Constant.Val.getAsString(Ctx, T),
                         printHex(Constant.Val.getInt()))
        .str();
  return Constant.Val.getAsString(Ctx, T);
}

llvm::Optional<std::string> printExprValue(const SelectionTree::Node *N,
                                           const ASTContext &Ctx) {
  for (; N; N = N->Parent) {
    // Try to evaluate the first evaluatable enclosing expression.
    if (const Expr *E = N->ASTNode.get<Expr>()) {
      // Once we cross an expression of type 'cv void', the evaluated result
      // has nothing to do with our original cursor position.
      if (!E->getType().isNull() && E->getType()->isVoidType())
        break;
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

llvm::Optional<StringRef> fieldName(const Expr *E) {
  const auto *ME = llvm::dyn_cast<MemberExpr>(E->IgnoreCasts());
  if (!ME || !llvm::isa<CXXThisExpr>(ME->getBase()->IgnoreCasts()))
    return llvm::None;
  const auto *Field = llvm::dyn_cast<FieldDecl>(ME->getMemberDecl());
  if (!Field || !Field->getDeclName().isIdentifier())
    return llvm::None;
  return Field->getDeclName().getAsIdentifierInfo()->getName();
}

// If CMD is of the form T foo() { return FieldName; } then returns "FieldName".
llvm::Optional<StringRef> getterVariableName(const CXXMethodDecl *CMD) {
  assert(CMD->hasBody());
  if (CMD->getNumParams() != 0 || CMD->isVariadic())
    return llvm::None;
  const auto *Body = llvm::dyn_cast<CompoundStmt>(CMD->getBody());
  const auto *OnlyReturn = (Body && Body->size() == 1)
                               ? llvm::dyn_cast<ReturnStmt>(Body->body_front())
                               : nullptr;
  if (!OnlyReturn || !OnlyReturn->getRetValue())
    return llvm::None;
  return fieldName(OnlyReturn->getRetValue());
}

// If CMD is one of the forms:
//   void foo(T arg) { FieldName = arg; }
//   R foo(T arg) { FieldName = arg; return *this; }
//   void foo(T arg) { FieldName = std::move(arg); }
//   R foo(T arg) { FieldName = std::move(arg); return *this; }
// then returns "FieldName"
llvm::Optional<StringRef> setterVariableName(const CXXMethodDecl *CMD) {
  assert(CMD->hasBody());
  if (CMD->isConst() || CMD->getNumParams() != 1 || CMD->isVariadic())
    return llvm::None;
  const ParmVarDecl *Arg = CMD->getParamDecl(0);
  if (Arg->isParameterPack())
    return llvm::None;

  const auto *Body = llvm::dyn_cast<CompoundStmt>(CMD->getBody());
  if (!Body || Body->size() == 0 || Body->size() > 2)
    return llvm::None;
  // If the second statement exists, it must be `return this` or `return *this`.
  if (Body->size() == 2) {
    auto *Ret = llvm::dyn_cast<ReturnStmt>(Body->body_back());
    if (!Ret || !Ret->getRetValue())
      return llvm::None;
    const Expr *RetVal = Ret->getRetValue()->IgnoreCasts();
    if (const auto *UO = llvm::dyn_cast<UnaryOperator>(RetVal)) {
      if (UO->getOpcode() != UO_Deref)
        return llvm::None;
      RetVal = UO->getSubExpr()->IgnoreCasts();
    }
    if (!llvm::isa<CXXThisExpr>(RetVal))
      return llvm::None;
  }
  // The first statement must be an assignment of the arg to a field.
  const Expr *LHS, *RHS;
  if (const auto *BO = llvm::dyn_cast<BinaryOperator>(Body->body_front())) {
    if (BO->getOpcode() != BO_Assign)
      return llvm::None;
    LHS = BO->getLHS();
    RHS = BO->getRHS();
  } else if (const auto *COCE =
                 llvm::dyn_cast<CXXOperatorCallExpr>(Body->body_front())) {
    if (COCE->getOperator() != OO_Equal || COCE->getNumArgs() != 2)
      return llvm::None;
    LHS = COCE->getArg(0);
    RHS = COCE->getArg(1);
  } else {
    return llvm::None;
  }

  // Detect the case when the item is moved into the field.
  if (auto *CE = llvm::dyn_cast<CallExpr>(RHS->IgnoreCasts())) {
    if (CE->getNumArgs() != 1)
      return llvm::None;
    auto *ND = llvm::dyn_cast_or_null<NamedDecl>(CE->getCalleeDecl());
    if (!ND || !ND->getIdentifier() || ND->getName() != "move" ||
        !ND->isInStdNamespace())
      return llvm::None;
    RHS = CE->getArg(0);
  }

  auto *DRE = llvm::dyn_cast<DeclRefExpr>(RHS->IgnoreCasts());
  if (!DRE || DRE->getDecl() != Arg)
    return llvm::None;
  return fieldName(LHS);
}

std::string synthesizeDocumentation(const NamedDecl *ND) {
  if (const auto *CMD = llvm::dyn_cast<CXXMethodDecl>(ND)) {
    // Is this an ordinary, non-static method whose definition is visible?
    if (CMD->getDeclName().isIdentifier() && !CMD->isStatic() &&
        (CMD = llvm::dyn_cast_or_null<CXXMethodDecl>(CMD->getDefinition())) &&
        CMD->hasBody()) {
      if (const auto GetterField = getterVariableName(CMD))
        return llvm::formatv("Trivial accessor for `{0}`.", *GetterField);
      if (const auto SetterField = setterVariableName(CMD))
        return llvm::formatv("Trivial setter for `{0}`.", *SetterField);
    }
  }
  return "";
}

/// Generate a \p Hover object given the declaration \p D.
HoverInfo getHoverContents(const NamedDecl *D, const PrintingPolicy &PP,
                           const SymbolIndex *Index,
                           const syntax::TokenBuffer &TB) {
  HoverInfo HI;
  auto &Ctx = D->getASTContext();

  HI.AccessSpecifier = getAccessSpelling(D->getAccess()).str();
  HI.NamespaceScope = getNamespaceScope(D);
  if (!HI.NamespaceScope->empty())
    HI.NamespaceScope->append("::");
  HI.LocalScope = getLocalScope(D);
  if (!HI.LocalScope.empty())
    HI.LocalScope.append("::");

  HI.Name = printName(Ctx, *D);
  const auto *CommentD = getDeclForComment(D);
  HI.Documentation = getDeclComment(Ctx, *CommentD);
  enhanceFromIndex(HI, *CommentD, Index);
  if (HI.Documentation.empty())
    HI.Documentation = synthesizeDocumentation(D);

  HI.Kind = index::getSymbolInfo(D).Kind;

  // Fill in template params.
  if (const TemplateDecl *TD = D->getDescribedTemplate()) {
    HI.TemplateParameters =
        fetchTemplateParameters(TD->getTemplateParameters(), PP);
    D = TD;
  } else if (const FunctionDecl *FD = D->getAsFunction()) {
    if (const auto *FTD = FD->getDescribedTemplate()) {
      HI.TemplateParameters =
          fetchTemplateParameters(FTD->getTemplateParameters(), PP);
      D = FTD;
    }
  }

  // Fill in types and params.
  if (const FunctionDecl *FD = getUnderlyingFunction(D))
    fillFunctionTypeAndParams(HI, D, FD, PP);
  else if (const auto *VD = dyn_cast<ValueDecl>(D))
    HI.Type = printType(VD->getType(), Ctx, PP);
  else if (const auto *TTP = dyn_cast<TemplateTypeParmDecl>(D))
    HI.Type = TTP->wasDeclaredWithTypename() ? "typename" : "class";
  else if (const auto *TTP = dyn_cast<TemplateTemplateParmDecl>(D))
    HI.Type = printType(TTP, PP);
  else if (const auto *VT = dyn_cast<VarTemplateDecl>(D))
    HI.Type = printType(VT->getTemplatedDecl()->getType(), Ctx, PP);
  else if (const auto *TN = dyn_cast<TypedefNameDecl>(D))
    HI.Type = printType(TN->getUnderlyingType().getDesugaredType(Ctx), Ctx, PP);
  else if (const auto *TAT = dyn_cast<TypeAliasTemplateDecl>(D))
    HI.Type = printType(TAT->getTemplatedDecl()->getUnderlyingType(), Ctx, PP);

  // Fill in value with evaluated initializer if possible.
  if (const auto *Var = dyn_cast<VarDecl>(D)) {
    if (const Expr *Init = Var->getInit())
      HI.Value = printExprValue(Init, Ctx);
  } else if (const auto *ECD = dyn_cast<EnumConstantDecl>(D)) {
    // Dependent enums (e.g. nested in template classes) don't have values yet.
    if (!ECD->getType()->isDependentType())
      HI.Value = toString(ECD->getInitVal(), 10);
  }

  HI.Definition = printDefinition(D, PP, TB);
  return HI;
}

/// Generate a \p Hover object given the macro \p MacroDecl.
HoverInfo getHoverContents(const DefinedMacro &Macro, ParsedAST &AST) {
  HoverInfo HI;
  SourceManager &SM = AST.getSourceManager();
  HI.Name = std::string(Macro.Name);
  HI.Kind = index::SymbolKind::Macro;
  // FIXME: Populate documentation
  // FIXME: Populate parameters

  // Try to get the full definition, not just the name
  SourceLocation StartLoc = Macro.Info->getDefinitionLoc();
  SourceLocation EndLoc = Macro.Info->getDefinitionEndLoc();
  // Ensure that EndLoc is a valid offset. For example it might come from
  // preamble, and source file might've changed, in such a scenario EndLoc still
  // stays valid, but getLocForEndOfToken will fail as it is no longer a valid
  // offset.
  // Note that this check is just to ensure there's text data inside the range.
  // It will still succeed even when the data inside the range is irrelevant to
  // macro definition.
  if (SM.getPresumedLoc(EndLoc, /*UseLineDirectives=*/false).isValid()) {
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

std::string typeAsDefinition(const HoverInfo::PrintedType &PType) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << PType.Type;
  if (PType.AKA)
    OS << " // aka: " << *PType.AKA;
  OS.flush();
  return Result;
}

llvm::Optional<HoverInfo> getThisExprHoverContents(const CXXThisExpr *CTE,
                                                   ASTContext &ASTCtx,
                                                   const PrintingPolicy &PP) {
  QualType OriginThisType = CTE->getType()->getPointeeType();
  QualType ClassType = declaredType(OriginThisType->getAsTagDecl());
  // For partial specialization class, origin `this` pointee type will be
  // parsed as `InjectedClassNameType`, which will ouput template arguments
  // like "type-parameter-0-0". So we retrieve user written class type in this
  // case.
  QualType PrettyThisType = ASTCtx.getPointerType(
      QualType(ClassType.getTypePtr(), OriginThisType.getCVRQualifiers()));

  HoverInfo HI;
  HI.Name = "this";
  HI.Definition = typeAsDefinition(printType(PrettyThisType, ASTCtx, PP));
  return HI;
}

/// Generate a HoverInfo object given the deduced type \p QT
HoverInfo getDeducedTypeHoverContents(QualType QT, const syntax::Token &Tok,
                                      ASTContext &ASTCtx,
                                      const PrintingPolicy &PP,
                                      const SymbolIndex *Index) {
  HoverInfo HI;
  // FIXME: distinguish decltype(auto) vs decltype(expr)
  HI.Name = tok::getTokenName(Tok.kind());
  HI.Kind = index::SymbolKind::TypeAlias;

  if (QT->isUndeducedAutoType()) {
    HI.Definition = "/* not deduced */";
  } else {
    HI.Definition = typeAsDefinition(printType(QT, ASTCtx, PP));

    if (const auto *D = QT->getAsTagDecl()) {
      const auto *CommentD = getDeclForComment(D);
      HI.Documentation = getDeclComment(ASTCtx, *CommentD);
      enhanceFromIndex(HI, *CommentD, Index);
    }
  }

  return HI;
}

bool isLiteral(const Expr *E) {
  // Unfortunately there's no common base Literal classes inherits from
  // (apart from Expr), therefore these exclusions.
  return llvm::isa<CompoundLiteralExpr>(E) ||
         llvm::isa<CXXBoolLiteralExpr>(E) ||
         llvm::isa<CXXNullPtrLiteralExpr>(E) ||
         llvm::isa<FixedPointLiteral>(E) || llvm::isa<FloatingLiteral>(E) ||
         llvm::isa<ImaginaryLiteral>(E) || llvm::isa<IntegerLiteral>(E) ||
         llvm::isa<StringLiteral>(E) || llvm::isa<UserDefinedLiteral>(E);
}

llvm::StringLiteral getNameForExpr(const Expr *E) {
  // FIXME: Come up with names for `special` expressions.
  //
  // It's an known issue for GCC5, https://godbolt.org/z/Z_tbgi. Work around
  // that by using explicit conversion constructor.
  //
  // TODO: Once GCC5 is fully retired and not the minimal requirement as stated
  // in `GettingStarted`, please remove the explicit conversion constructor.
  return llvm::StringLiteral("expression");
}

// Generates hover info for `this` and evaluatable expressions.
// FIXME: Support hover for literals (esp user-defined)
llvm::Optional<HoverInfo> getHoverContents(const Expr *E, ParsedAST &AST,
                                           const PrintingPolicy &PP,
                                           const SymbolIndex *Index) {
  // There's not much value in hovering over "42" and getting a hover card
  // saying "42 is an int", similar for other literals.
  if (isLiteral(E))
    return llvm::None;

  HoverInfo HI;
  // For `this` expr we currently generate hover with pointee type.
  if (const CXXThisExpr *CTE = dyn_cast<CXXThisExpr>(E))
    return getThisExprHoverContents(CTE, AST.getASTContext(), PP);
  // For expressions we currently print the type and the value, iff it is
  // evaluatable.
  if (auto Val = printExprValue(E, AST.getASTContext())) {
    HI.Type = printType(E->getType(), AST.getASTContext(), PP);
    HI.Value = *Val;
    HI.Name = std::string(getNameForExpr(E));
    return HI;
  }
  return llvm::None;
}

// Generates hover info for attributes.
llvm::Optional<HoverInfo> getHoverContents(const Attr *A, ParsedAST &AST) {
  HoverInfo HI;
  HI.Name = A->getSpelling();
  if (A->hasScope())
    HI.LocalScope = A->getScopeName()->getName().str();
  {
    llvm::raw_string_ostream OS(HI.Definition);
    A->printPretty(OS, AST.getASTContext().getPrintingPolicy());
  }
  HI.Documentation = Attr::getDocumentation(A->getKind()).str();
  return HI;
}

bool isParagraphBreak(llvm::StringRef Rest) {
  return Rest.ltrim(" \t").startswith("\n");
}

bool punctuationIndicatesLineBreak(llvm::StringRef Line) {
  constexpr llvm::StringLiteral Punctuation = R"txt(.:,;!?)txt";

  Line = Line.rtrim();
  return !Line.empty() && Punctuation.contains(Line.back());
}

bool isHardLineBreakIndicator(llvm::StringRef Rest) {
  // '-'/'*' md list, '@'/'\' documentation command, '>' md blockquote,
  // '#' headings, '`' code blocks
  constexpr llvm::StringLiteral LinebreakIndicators = R"txt(-*@\>#`)txt";

  Rest = Rest.ltrim(" \t");
  if (Rest.empty())
    return false;

  if (LinebreakIndicators.contains(Rest.front()))
    return true;

  if (llvm::isDigit(Rest.front())) {
    llvm::StringRef AfterDigit = Rest.drop_while(llvm::isDigit);
    if (AfterDigit.startswith(".") || AfterDigit.startswith(")"))
      return true;
  }
  return false;
}

bool isHardLineBreakAfter(llvm::StringRef Line, llvm::StringRef Rest) {
  // Should we also consider whether Line is short?
  return punctuationIndicatesLineBreak(Line) || isHardLineBreakIndicator(Rest);
}

void addLayoutInfo(const NamedDecl &ND, HoverInfo &HI) {
  if (ND.isInvalidDecl())
    return;

  const auto &Ctx = ND.getASTContext();
  if (auto *RD = llvm::dyn_cast<RecordDecl>(&ND)) {
    if (auto Size = Ctx.getTypeSizeInCharsIfKnown(RD->getTypeForDecl()))
      HI.Size = Size->getQuantity();
    return;
  }

  if (const auto *FD = llvm::dyn_cast<FieldDecl>(&ND)) {
    const auto *Record = FD->getParent();
    if (Record)
      Record = Record->getDefinition();
    if (Record && !Record->isInvalidDecl() && !Record->isDependentType() &&
        !FD->isBitField()) {
      const ASTRecordLayout &Layout = Ctx.getASTRecordLayout(Record);
      HI.Offset = Layout.getFieldOffset(FD->getFieldIndex()) / 8;
      if (auto Size = Ctx.getTypeSizeInCharsIfKnown(FD->getType())) {
        HI.Size = FD->isZeroSize(Ctx) ? 0 : Size->getQuantity();
        unsigned EndOfField = *HI.Offset + *HI.Size;

        // Calculate padding following the field.
        if (!Record->isUnion() &&
            FD->getFieldIndex() + 1 < Layout.getFieldCount()) {
          // Measure padding up to the next class field.
          unsigned NextOffset =
              Layout.getFieldOffset(FD->getFieldIndex() + 1) / 8;
          if (NextOffset >= EndOfField) // next field could be a bitfield!
            HI.Padding = NextOffset - EndOfField;
        } else {
          // Measure padding up to the end of the object.
          HI.Padding = Layout.getSize().getQuantity() - EndOfField;
        }
      }
      // Offset in a union is always zero, so not really useful to report.
      if (Record->isUnion())
        HI.Offset.reset();
    }
    return;
  }
}

// If N is passed as argument to a function, fill HI.CalleeArgInfo with
// information about that argument.
void maybeAddCalleeArgInfo(const SelectionTree::Node *N, HoverInfo &HI,
                           const PrintingPolicy &PP) {
  const auto &OuterNode = N->outerImplicit();
  if (!OuterNode.Parent)
    return;
  const auto *CE = OuterNode.Parent->ASTNode.get<CallExpr>();
  if (!CE)
    return;
  const FunctionDecl *FD = CE->getDirectCallee();
  // For non-function-call-like operatators (e.g. operator+, operator<<) it's
  // not immediattely obvious what the "passed as" would refer to and, given
  // fixed function signature, the value would be very low anyway, so we choose
  // to not support that.
  // Both variadic functions and operator() (especially relevant for lambdas)
  // should be supported in the future.
  if (!FD || FD->isOverloadedOperator() || FD->isVariadic())
    return;

  // Find argument index for N.
  for (unsigned I = 0; I < CE->getNumArgs() && I < FD->getNumParams(); ++I) {
    if (CE->getArg(I) != OuterNode.ASTNode.get<Expr>())
      continue;

    // Extract matching argument from function declaration.
    if (const ParmVarDecl *PVD = FD->getParamDecl(I))
      HI.CalleeArgInfo.emplace(toHoverInfoParam(PVD, PP));
    break;
  }
  if (!HI.CalleeArgInfo)
    return;

  // If we found a matching argument, also figure out if it's a
  // [const-]reference. For this we need to walk up the AST from the arg itself
  // to CallExpr and check all implicit casts, constructor calls, etc.
  HoverInfo::PassType PassType;
  if (const auto *E = N->ASTNode.get<Expr>()) {
    if (E->getType().isConstQualified())
      PassType.PassBy = HoverInfo::PassType::ConstRef;
  }

  for (auto *CastNode = N->Parent;
       CastNode != OuterNode.Parent && !PassType.Converted;
       CastNode = CastNode->Parent) {
    if (const auto *ImplicitCast = CastNode->ASTNode.get<ImplicitCastExpr>()) {
      switch (ImplicitCast->getCastKind()) {
      case CK_NoOp:
      case CK_DerivedToBase:
      case CK_UncheckedDerivedToBase:
        // If it was a reference before, it's still a reference.
        if (PassType.PassBy != HoverInfo::PassType::Value)
          PassType.PassBy = ImplicitCast->getType().isConstQualified()
                                ? HoverInfo::PassType::ConstRef
                                : HoverInfo::PassType::Ref;
        break;
      case CK_LValueToRValue:
      case CK_ArrayToPointerDecay:
      case CK_FunctionToPointerDecay:
      case CK_NullToPointer:
      case CK_NullToMemberPointer:
        // No longer a reference, but we do not show this as type conversion.
        PassType.PassBy = HoverInfo::PassType::Value;
        break;
      default:
        PassType.PassBy = HoverInfo::PassType::Value;
        PassType.Converted = true;
        break;
      }
    } else if (const auto *CtorCall =
                   CastNode->ASTNode.get<CXXConstructExpr>()) {
      // We want to be smart about copy constructors. They should not show up as
      // type conversion, but instead as passing by value.
      if (CtorCall->getConstructor()->isCopyConstructor())
        PassType.PassBy = HoverInfo::PassType::Value;
      else
        PassType.Converted = true;
    } else { // Unknown implicit node, assume type conversion.
      PassType.PassBy = HoverInfo::PassType::Value;
      PassType.Converted = true;
    }
  }

  HI.CallPassType.emplace(PassType);
}

} // namespace

llvm::Optional<HoverInfo> getHover(ParsedAST &AST, Position Pos,
                                   const format::FormatStyle &Style,
                                   const SymbolIndex *Index) {
  PrintingPolicy PP =
      getPrintingPolicy(AST.getASTContext().getPrintingPolicy());
  const SourceManager &SM = AST.getSourceManager();
  auto CurLoc = sourceLocationInMainFile(SM, Pos);
  if (!CurLoc) {
    llvm::consumeError(CurLoc.takeError());
    return llvm::None;
  }
  const auto &TB = AST.getTokens();
  auto TokensTouchingCursor = syntax::spelledTokensTouching(*CurLoc, TB);
  // Early exit if there were no tokens around the cursor.
  if (TokensTouchingCursor.empty())
    return llvm::None;

  // Show full header file path if cursor is on include directive.
  if (const auto MainFilePath =
          getCanonicalPath(SM.getFileEntryForID(SM.getMainFileID()), SM)) {
    for (const auto &Inc : AST.getIncludeStructure().MainFileIncludes) {
      if (Inc.Resolved.empty() || Inc.HashLine != Pos.line)
        continue;
      HoverInfo HI;
      HI.Name = std::string(llvm::sys::path::filename(Inc.Resolved));
      // FIXME: We don't have a fitting value for Kind.
      HI.Definition =
          URIForFile::canonicalize(Inc.Resolved, *MainFilePath).file().str();
      HI.DefinitionLanguage = "";
      return HI;
    }
  }

  // To be used as a backup for highlighting the selected token, we use back as
  // it aligns better with biases elsewhere (editors tend to send the position
  // for the left of the hovered token).
  CharSourceRange HighlightRange =
      TokensTouchingCursor.back().range(SM).toCharRange(SM);
  llvm::Optional<HoverInfo> HI;
  // Macros and deducedtype only works on identifiers and auto/decltype keywords
  // respectively. Therefore they are only trggered on whichever works for them,
  // similar to SelectionTree::create().
  for (const auto &Tok : TokensTouchingCursor) {
    if (Tok.kind() == tok::identifier) {
      // Prefer the identifier token as a fallback highlighting range.
      HighlightRange = Tok.range(SM).toCharRange(SM);
      if (auto M = locateMacroAt(Tok, AST.getPreprocessor())) {
        HI = getHoverContents(*M, AST);
        break;
      }
    } else if (Tok.kind() == tok::kw_auto || Tok.kind() == tok::kw_decltype) {
      if (auto Deduced = getDeducedType(AST.getASTContext(), Tok.location())) {
        HI = getDeducedTypeHoverContents(*Deduced, Tok, AST.getASTContext(), PP,
                                         Index);
        HighlightRange = Tok.range(SM).toCharRange(SM);
        break;
      }

      // If we can't find interesting hover information for this
      // auto/decltype keyword, return nothing to avoid showing
      // irrelevant or incorrect informations.
      return llvm::None;
    }
  }

  // If it wasn't auto/decltype or macro, look for decls and expressions.
  if (!HI) {
    auto Offset = SM.getFileOffset(*CurLoc);
    // Editors send the position on the left of the hovered character.
    // So our selection tree should be biased right. (Tested with VSCode).
    SelectionTree ST =
        SelectionTree::createRight(AST.getASTContext(), TB, Offset, Offset);
    if (const SelectionTree::Node *N = ST.commonAncestor()) {
      // FIXME: Fill in HighlightRange with range coming from N->ASTNode.
      auto Decls = explicitReferenceTargets(N->ASTNode, DeclRelation::Alias,
                                            AST.getHeuristicResolver());
      if (!Decls.empty()) {
        HI = getHoverContents(Decls.front(), PP, Index, TB);
        // Layout info only shown when hovering on the field/class itself.
        if (Decls.front() == N->ASTNode.get<Decl>())
          addLayoutInfo(*Decls.front(), *HI);
        // Look for a close enclosing expression to show the value of.
        if (!HI->Value)
          HI->Value = printExprValue(N, AST.getASTContext());
        maybeAddCalleeArgInfo(N, *HI, PP);
      } else if (const Expr *E = N->ASTNode.get<Expr>()) {
        HI = getHoverContents(E, AST, PP, Index);
      } else if (const Attr *A = N->ASTNode.get<Attr>()) {
        HI = getHoverContents(A, AST);
      }
      // FIXME: support hovers for other nodes?
      //  - built-in types
    }
  }

  if (!HI)
    return llvm::None;

  auto Replacements = format::reformat(
      Style, HI->Definition, tooling::Range(0, HI->Definition.size()));
  if (auto Formatted =
          tooling::applyAllReplacements(HI->Definition, Replacements))
    HI->Definition = *Formatted;
  HI->DefinitionLanguage = getMarkdownLanguage(AST.getASTContext());
  HI->SymRange = halfOpenToRange(SM, HighlightRange);

  return HI;
}

markup::Document HoverInfo::present() const {
  markup::Document Output;

  // Header contains a text of the form:
  // variable `var`
  //
  // class `X`
  //
  // function `foo`
  //
  // expression
  //
  // Note that we are making use of a level-3 heading because VSCode renders
  // level 1 and 2 headers in a huge font, see
  // https://github.com/microsoft/vscode/issues/88417 for details.
  markup::Paragraph &Header = Output.addHeading(3);
  if (Kind != index::SymbolKind::Unknown)
    Header.appendText(index::getSymbolKindString(Kind)).appendSpace();
  assert(!Name.empty() && "hover triggered on a nameless symbol");
  Header.appendCode(Name);

  // Put a linebreak after header to increase readability.
  Output.addRuler();
  // Print Types on their own lines to reduce chances of getting line-wrapped by
  // editor, as they might be long.
  if (ReturnType) {
    // For functions we display signature in a list form, e.g.:
    // → `x`
    // Parameters:
    // - `bool param1`
    // - `int param2 = 5`
    Output.addParagraph().appendText("→ ").appendCode(
        llvm::to_string(*ReturnType));
  }

  if (Parameters && !Parameters->empty()) {
    Output.addParagraph().appendText("Parameters: ");
    markup::BulletList &L = Output.addBulletList();
    for (const auto &Param : *Parameters)
      L.addItem().addParagraph().appendCode(llvm::to_string(Param));
  }

  // Don't print Type after Parameters or ReturnType as this will just duplicate
  // the information
  if (Type && !ReturnType && !Parameters)
    Output.addParagraph().appendText("Type: ").appendCode(
        llvm::to_string(*Type));

  if (Value) {
    markup::Paragraph &P = Output.addParagraph();
    P.appendText("Value = ");
    P.appendCode(*Value);
  }

  if (Offset)
    Output.addParagraph().appendText(
        llvm::formatv("Offset: {0} byte{1}", *Offset, *Offset == 1 ? "" : "s")
            .str());
  if (Size) {
    auto &P = Output.addParagraph().appendText(
        llvm::formatv("Size: {0} byte{1}", *Size, *Size == 1 ? "" : "s").str());
    if (Padding && *Padding != 0)
      P.appendText(llvm::formatv(" (+{0} padding)", *Padding).str());
  }

  if (CalleeArgInfo) {
    assert(CallPassType);
    std::string Buffer;
    llvm::raw_string_ostream OS(Buffer);
    OS << "Passed ";
    if (CallPassType->PassBy != HoverInfo::PassType::Value) {
      OS << "by ";
      if (CallPassType->PassBy == HoverInfo::PassType::ConstRef)
        OS << "const ";
      OS << "reference ";
    }
    if (CalleeArgInfo->Name)
      OS << "as " << CalleeArgInfo->Name;
    if (CallPassType->Converted && CalleeArgInfo->Type)
      OS << " (converted to " << CalleeArgInfo->Type->Type << ")";
    Output.addParagraph().appendText(OS.str());
  }

  if (!Documentation.empty())
    parseDocumentation(Documentation, Output);

  if (!Definition.empty()) {
    Output.addRuler();
    std::string ScopeComment;
    // Drop trailing "::".
    if (!LocalScope.empty()) {
      // Container name, e.g. class, method, function.
      // We might want to propagate some info about container type to print
      // function foo, class X, method X::bar, etc.
      ScopeComment =
          "// In " + llvm::StringRef(LocalScope).rtrim(':').str() + '\n';
    } else if (NamespaceScope && !NamespaceScope->empty()) {
      ScopeComment = "// In namespace " +
                     llvm::StringRef(*NamespaceScope).rtrim(':').str() + '\n';
    }
    std::string DefinitionWithAccess = !AccessSpecifier.empty()
                                           ? AccessSpecifier + ": " + Definition
                                           : Definition;
    // Note that we don't print anything for global namespace, to not annoy
    // non-c++ projects or projects that are not making use of namespaces.
    Output.addCodeBlock(ScopeComment + DefinitionWithAccess,
                        DefinitionLanguage);
  }

  return Output;
}

// If the backtick at `Offset` starts a probable quoted range, return the range
// (including the quotes).
llvm::Optional<llvm::StringRef> getBacktickQuoteRange(llvm::StringRef Line,
                                                      unsigned Offset) {
  assert(Line[Offset] == '`');

  // The open-quote is usually preceded by whitespace.
  llvm::StringRef Prefix = Line.substr(0, Offset);
  constexpr llvm::StringLiteral BeforeStartChars = " \t(=";
  if (!Prefix.empty() && !BeforeStartChars.contains(Prefix.back()))
    return llvm::None;

  // The quoted string must be nonempty and usually has no leading/trailing ws.
  auto Next = Line.find('`', Offset + 1);
  if (Next == llvm::StringRef::npos)
    return llvm::None;
  llvm::StringRef Contents = Line.slice(Offset + 1, Next);
  if (Contents.empty() || isWhitespace(Contents.front()) ||
      isWhitespace(Contents.back()))
    return llvm::None;

  // The close-quote is usually followed by whitespace or punctuation.
  llvm::StringRef Suffix = Line.substr(Next + 1);
  constexpr llvm::StringLiteral AfterEndChars = " \t)=.,;:";
  if (!Suffix.empty() && !AfterEndChars.contains(Suffix.front()))
    return llvm::None;

  return Line.slice(Offset, Next + 1);
}

void parseDocumentationLine(llvm::StringRef Line, markup::Paragraph &Out) {
  // Probably this is appendText(Line), but scan for something interesting.
  for (unsigned I = 0; I < Line.size(); ++I) {
    switch (Line[I]) {
    case '`':
      if (auto Range = getBacktickQuoteRange(Line, I)) {
        Out.appendText(Line.substr(0, I));
        Out.appendCode(Range->trim("`"), /*Preserve=*/true);
        return parseDocumentationLine(Line.substr(I + Range->size()), Out);
      }
      break;
    }
  }
  Out.appendText(Line).appendSpace();
}

void parseDocumentation(llvm::StringRef Input, markup::Document &Output) {
  std::vector<llvm::StringRef> ParagraphLines;
  auto FlushParagraph = [&] {
    if (ParagraphLines.empty())
      return;
    auto &P = Output.addParagraph();
    for (llvm::StringRef Line : ParagraphLines)
      parseDocumentationLine(Line, P);
    ParagraphLines.clear();
  };

  llvm::StringRef Line, Rest;
  for (std::tie(Line, Rest) = Input.split('\n');
       !(Line.empty() && Rest.empty());
       std::tie(Line, Rest) = Rest.split('\n')) {

    // After a linebreak remove spaces to avoid 4 space markdown code blocks.
    // FIXME: make FlushParagraph handle this.
    Line = Line.ltrim();
    if (!Line.empty())
      ParagraphLines.push_back(Line);

    if (isParagraphBreak(Rest) || isHardLineBreakAfter(Line, Rest)) {
      FlushParagraph();
    }
  }
  FlushParagraph();
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const HoverInfo::PrintedType &T) {
  OS << T.Type;
  if (T.AKA)
    OS << " (aka " << *T.AKA << ")";
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const HoverInfo::Param &P) {
  if (P.Type)
    OS << P.Type->Type;
  if (P.Name)
    OS << " " << *P.Name;
  if (P.Default)
    OS << " = " << *P.Default;
  if (P.Type && P.Type->AKA)
    OS << " (aka " << *P.Type->AKA << ")";
  return OS;
}

} // namespace clangd
} // namespace clang
