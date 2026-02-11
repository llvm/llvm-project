//===--- HLSLBuiltinTypeDeclBuilder.cpp - HLSL Builtin Type Decl Builder --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper classes for creating HLSL builtin class types. Used by external HLSL
// sema source.
//
//===----------------------------------------------------------------------===//

#include "HLSLBuiltinTypeDeclBuilder.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/HLSLResource.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaHLSL.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm::hlsl;

namespace clang {

namespace hlsl {

namespace {

static FunctionDecl *lookupBuiltinFunction(Sema &S, StringRef Name) {
  IdentifierInfo &II =
      S.getASTContext().Idents.get(Name, tok::TokenKind::identifier);
  DeclarationNameInfo NameInfo =
      DeclarationNameInfo(DeclarationName(&II), SourceLocation());
  LookupResult R(S, NameInfo, Sema::LookupOrdinaryName);
  // AllowBuiltinCreation is false but LookupDirect will create
  // the builtin when searching the global scope anyways...
  S.LookupName(R, S.getCurScope());
  // FIXME: If the builtin function was user-declared in global scope,
  // this assert *will* fail. Should this call LookupBuiltin instead?
  assert(R.isSingleResult() &&
         "Since this is a builtin it should always resolve!");
  return cast<FunctionDecl>(R.getFoundDecl());
}

static QualType lookupBuiltinType(Sema &S, StringRef Name, DeclContext *DC) {
  IdentifierInfo &II =
      S.getASTContext().Idents.get(Name, tok::TokenKind::identifier);
  LookupResult Result(S, &II, SourceLocation(), Sema::LookupTagName);
  S.LookupQualifiedName(Result, DC);
  assert(!Result.empty() && "Builtin type not found");
  QualType Ty =
      S.getASTContext().getTypeDeclType(Result.getAsSingle<TypeDecl>());
  S.RequireCompleteType(SourceLocation(), Ty,
                        diag::err_tentative_def_incomplete_type);
  return Ty;
}

CXXConstructorDecl *lookupCopyConstructor(QualType ResTy) {
  assert(ResTy->isRecordType() && "not a CXXRecord type");
  for (auto *CD : ResTy->getAsCXXRecordDecl()->ctors())
    if (CD->isCopyConstructor())
      return CD;
  return nullptr;
}

ParameterABI
convertParamModifierToParamABI(HLSLParamModifierAttr::Spelling Modifier) {
  assert(Modifier != HLSLParamModifierAttr::Spelling::Keyword_in &&
         "HLSL 'in' parameters modifier cannot be converted to ParameterABI");
  switch (Modifier) {
  case HLSLParamModifierAttr::Spelling::Keyword_out:
    return ParameterABI::HLSLOut;
  case HLSLParamModifierAttr::Spelling::Keyword_inout:
    return ParameterABI::HLSLInOut;
  default:
    llvm_unreachable("Invalid HLSL parameter modifier");
  }
}

QualType getInoutParameterType(ASTContext &AST, QualType Ty) {
  assert(!Ty->isReferenceType() &&
         "Pointer and reference types cannot be inout or out parameters");
  Ty = AST.getLValueReferenceType(Ty);
  Ty.addRestrict();
  return Ty;
}

} // namespace

// Builder for template arguments of builtin types. Used internally
// by BuiltinTypeDeclBuilder.
struct TemplateParameterListBuilder {
  BuiltinTypeDeclBuilder &Builder;
  llvm::SmallVector<NamedDecl *> Params;

  TemplateParameterListBuilder(BuiltinTypeDeclBuilder &RB) : Builder(RB) {}
  ~TemplateParameterListBuilder();

  TemplateParameterListBuilder &
  addTypeParameter(StringRef Name, QualType DefaultValue = QualType());

  ConceptSpecializationExpr *
  constructConceptSpecializationExpr(Sema &S, ConceptDecl *CD);

  BuiltinTypeDeclBuilder &finalizeTemplateArgs(ConceptDecl *CD = nullptr);
};

// Builder for methods or constructors of builtin types. Allows creating methods
// or constructors of builtin types using the builder pattern like this:
//
//   BuiltinTypeMethodBuilder(RecordBuilder, "MethodName", ReturnType)
//       .addParam("param_name", Type, InOutModifier)
//       .callBuiltin("builtin_name", BuiltinParams...)
//       .finalize();
//
// The builder needs to have all of the parameters before it can create
// a CXXMethodDecl or CXXConstructorDecl. It collects them in addParam calls and
// when a first method that builds the body is called or when access to 'this`
// is needed it creates the CXXMethodDecl/CXXConstructorDecl and ParmVarDecls
// instances. These can then be referenced from the body building methods.
// Destructor or an explicit call to finalize() will complete the method
// definition.
//
// The callBuiltin helper method accepts constants via `Expr *` or placeholder
// value arguments to indicate which function arguments to forward to the
// builtin.
//
// If the method that is being built has a non-void return type the
// finalize() will create a return statement with the value of the last
// statement (unless the last statement is already a ReturnStmt or the return
// value is void).
struct BuiltinTypeMethodBuilder {
private:
  struct Param {
    const IdentifierInfo &NameII;
    QualType Ty;
    HLSLParamModifierAttr::Spelling Modifier;
    Param(const IdentifierInfo &NameII, QualType Ty,
          HLSLParamModifierAttr::Spelling Modifier)
        : NameII(NameII), Ty(Ty), Modifier(Modifier) {}
  };

  struct LocalVar {
    StringRef Name;
    QualType Ty;
    VarDecl *Decl;
    LocalVar(StringRef Name, QualType Ty) : Name(Name), Ty(Ty), Decl(nullptr) {}
  };

  BuiltinTypeDeclBuilder &DeclBuilder;
  DeclarationName Name;
  QualType ReturnTy;
  // method or constructor declaration
  // (CXXConstructorDecl derives from CXXMethodDecl)
  CXXMethodDecl *Method;
  bool IsConst;
  bool IsCtor;
  StorageClass SC;
  llvm::SmallVector<Param> Params;
  llvm::SmallVector<Stmt *> StmtsList;
  TemplateParameterList *TemplateParams;
  llvm::SmallVector<NamedDecl *> TemplateParamDecls;

  // Argument placeholders, inspired by std::placeholder. These are the indices
  // of arguments to forward to `callBuiltin` and other method builder methods.
  // Additional special values are:
  //   Handle   - refers to the resource handle.
  //   LastStmt - refers to the last statement in the method body; referencing
  //              LastStmt will remove the statement from the method body since
  //              it will be linked from the new expression being constructed.
  enum class PlaceHolder {
    _0,
    _1,
    _2,
    _3,
    _4,
    _5,
    Handle = 128,
    CounterHandle,
    LastStmt
  };

  Expr *convertPlaceholder(PlaceHolder PH);
  Expr *convertPlaceholder(LocalVar &Var);
  Expr *convertPlaceholder(Expr *E) { return E; }
  // Converts a QualType to an Expr that carries type information to builtins.
  Expr *convertPlaceholder(QualType Ty);

public:
  friend BuiltinTypeDeclBuilder;

  BuiltinTypeMethodBuilder(BuiltinTypeDeclBuilder &DB, DeclarationName &Name,
                           QualType ReturnTy, bool IsConst = false,
                           bool IsCtor = false, StorageClass SC = SC_None)
      : DeclBuilder(DB), Name(Name), ReturnTy(ReturnTy), Method(nullptr),
        IsConst(IsConst), IsCtor(IsCtor), SC(SC), TemplateParams(nullptr) {}

  BuiltinTypeMethodBuilder(BuiltinTypeDeclBuilder &DB, StringRef NameStr,
                           QualType ReturnTy, bool IsConst = false,
                           bool IsCtor = false, StorageClass SC = SC_None);
  BuiltinTypeMethodBuilder(const BuiltinTypeMethodBuilder &Other) = delete;

  ~BuiltinTypeMethodBuilder() { finalize(); }

  BuiltinTypeMethodBuilder &
  operator=(const BuiltinTypeMethodBuilder &Other) = delete;

  BuiltinTypeMethodBuilder &addParam(StringRef Name, QualType Ty,
                                     HLSLParamModifierAttr::Spelling Modifier =
                                         HLSLParamModifierAttr::Keyword_in);
  QualType addTemplateTypeParam(StringRef Name);
  BuiltinTypeMethodBuilder &declareLocalVar(LocalVar &Var);
  template <typename... Ts>
  BuiltinTypeMethodBuilder &callBuiltin(StringRef BuiltinName,
                                        QualType ReturnType, Ts &&...ArgSpecs);
  template <typename TLHS, typename TRHS>
  BuiltinTypeMethodBuilder &assign(TLHS LHS, TRHS RHS);
  template <typename T> BuiltinTypeMethodBuilder &dereference(T Ptr);

  template <typename T>
  BuiltinTypeMethodBuilder &accessHandleFieldOnResource(T ResourceRecord);
  template <typename ResourceT, typename ValueT>
  BuiltinTypeMethodBuilder &setHandleFieldOnResource(ResourceT ResourceRecord,
                                                     ValueT HandleValue);
  template <typename T>
  BuiltinTypeMethodBuilder &
  accessCounterHandleFieldOnResource(T ResourceRecord);
  template <typename ResourceT, typename ValueT>
  BuiltinTypeMethodBuilder &
  setCounterHandleFieldOnResource(ResourceT ResourceRecord, ValueT HandleValue);
  template <typename T> BuiltinTypeMethodBuilder &returnValue(T ReturnValue);
  BuiltinTypeMethodBuilder &returnThis();
  BuiltinTypeDeclBuilder &finalize();
  Expr *getResourceHandleExpr();
  Expr *getResourceCounterHandleExpr();

private:
  void createDecl();

  // Makes sure the declaration is created; should be called before any
  // statement added to the body or when access to 'this' is needed.
  void ensureCompleteDecl() {
    if (!Method)
      createDecl();
  }

  template <typename ResourceT, typename ValueT>
  BuiltinTypeMethodBuilder &setFieldOnResource(ResourceT ResourceRecord,
                                               ValueT HandleValue,
                                               FieldDecl *HandleField);
};

TemplateParameterListBuilder::~TemplateParameterListBuilder() {
  finalizeTemplateArgs();
}

TemplateParameterListBuilder &
TemplateParameterListBuilder::addTypeParameter(StringRef Name,
                                               QualType DefaultValue) {
  assert(!Builder.Record->isCompleteDefinition() &&
         "record is already complete");
  ASTContext &AST = Builder.SemaRef.getASTContext();
  unsigned Position = static_cast<unsigned>(Params.size());
  auto *Decl = TemplateTypeParmDecl::Create(
      AST, Builder.Record->getDeclContext(), SourceLocation(), SourceLocation(),
      /* TemplateDepth */ 0, Position,
      &AST.Idents.get(Name, tok::TokenKind::identifier),
      /* Typename */ true,
      /* ParameterPack */ false,
      /* HasTypeConstraint*/ false);
  if (!DefaultValue.isNull())
    Decl->setDefaultArgument(AST,
                             Builder.SemaRef.getTrivialTemplateArgumentLoc(
                                 DefaultValue, QualType(), SourceLocation()));

  Params.emplace_back(Decl);
  return *this;
}

// The concept specialization expression (CSE) constructed in
// constructConceptSpecializationExpr is constructed so that it
// matches the CSE that is constructed when parsing the below C++ code:
//
// template<typename T>
// concept is_typed_resource_element_compatible =
// __builtin_hlsl_typed_resource_element_compatible<T>
//
// template<typename element_type> requires
// is_typed_resource_element_compatible<element_type>
// struct RWBuffer {
//     element_type Val;
// };
//
// int fn() {
//     RWBuffer<int> Buf;
// }
//
// When dumping the AST and filtering for "RWBuffer", the resulting AST
// structure is what we're trying to construct below, specifically the
// CSE portion.
ConceptSpecializationExpr *
TemplateParameterListBuilder::constructConceptSpecializationExpr(
    Sema &S, ConceptDecl *CD) {
  ASTContext &Context = S.getASTContext();
  SourceLocation Loc = Builder.Record->getBeginLoc();
  DeclarationNameInfo DNI(CD->getDeclName(), Loc);
  NestedNameSpecifierLoc NNSLoc;
  DeclContext *DC = Builder.Record->getDeclContext();
  TemplateArgumentListInfo TALI(Loc, Loc);

  // Assume that the concept decl has just one template parameter
  // This parameter should have been added when CD was constructed
  // in getTypedBufferConceptDecl
  assert(CD->getTemplateParameters()->size() == 1 &&
         "unexpected concept decl parameter count");
  TemplateTypeParmDecl *ConceptTTPD =
      dyn_cast<TemplateTypeParmDecl>(CD->getTemplateParameters()->getParam(0));

  // this TemplateTypeParmDecl is the template for the resource, and is
  // used to construct a template argumentthat will be used
  // to construct the ImplicitConceptSpecializationDecl
  TemplateTypeParmDecl *T = TemplateTypeParmDecl::Create(
      Context,                          // AST context
      Builder.Record->getDeclContext(), // DeclContext
      SourceLocation(), SourceLocation(),
      /*D=*/0,                    // Depth in the template parameter list
      /*P=*/0,                    // Position in the template parameter list
      /*Id=*/nullptr,             // Identifier for 'T'
      /*Typename=*/true,          // Indicates this is a 'typename' or 'class'
      /*ParameterPack=*/false,    // Not a parameter pack
      /*HasTypeConstraint=*/false // Has no type constraint
  );

  T->setDeclContext(DC);

  QualType ConceptTType = Context.getTypeDeclType(ConceptTTPD);

  // this is the 2nd template argument node, on which
  // the concept constraint is actually being applied: 'element_type'
  TemplateArgument ConceptTA = TemplateArgument(ConceptTType);

  QualType CSETType = Context.getTypeDeclType(T);

  // this is the 1st template argument node, which represents
  // the abstract type that a concept would refer to: 'T'
  TemplateArgument CSETA = TemplateArgument(CSETType);

  ImplicitConceptSpecializationDecl *ImplicitCSEDecl =
      ImplicitConceptSpecializationDecl::Create(
          Context, Builder.Record->getDeclContext(), Loc, {CSETA});

  // Constraint satisfaction is used to construct the
  // ConceptSpecailizationExpr, and represents the 2nd Template Argument,
  // located at the bottom of the sample AST above.
  const ConstraintSatisfaction CS(CD, {ConceptTA});
  TemplateArgumentLoc TAL =
      S.getTrivialTemplateArgumentLoc(ConceptTA, QualType(), SourceLocation());

  TALI.addArgument(TAL);
  const ASTTemplateArgumentListInfo *ATALI =
      ASTTemplateArgumentListInfo::Create(Context, TALI);

  // In the concept reference, ATALI is what adds the extra
  // TemplateArgument node underneath CSE
  ConceptReference *CR =
      ConceptReference::Create(Context, NNSLoc, Loc, DNI, CD, CD, ATALI);

  ConceptSpecializationExpr *CSE =
      ConceptSpecializationExpr::Create(Context, CR, ImplicitCSEDecl, &CS);

  return CSE;
}

BuiltinTypeDeclBuilder &
TemplateParameterListBuilder::finalizeTemplateArgs(ConceptDecl *CD) {
  if (Params.empty())
    return Builder;

  ASTContext &AST = Builder.SemaRef.Context;
  ConceptSpecializationExpr *CSE =
      CD ? constructConceptSpecializationExpr(Builder.SemaRef, CD) : nullptr;
  auto *ParamList = TemplateParameterList::Create(
      AST, SourceLocation(), SourceLocation(), Params, SourceLocation(), CSE);
  Builder.Template = ClassTemplateDecl::Create(
      AST, Builder.Record->getDeclContext(), SourceLocation(),
      DeclarationName(Builder.Record->getIdentifier()), ParamList,
      Builder.Record);

  Builder.Record->setDescribedClassTemplate(Builder.Template);
  Builder.Template->setImplicit(true);
  Builder.Template->setLexicalDeclContext(Builder.Record->getDeclContext());

  // NOTE: setPreviousDecl before addDecl so new decl replace old decl when
  // make visible.
  Builder.Template->setPreviousDecl(Builder.PrevTemplate);
  Builder.Record->getDeclContext()->addDecl(Builder.Template);
  Params.clear();

  return Builder;
}

Expr *BuiltinTypeMethodBuilder::convertPlaceholder(PlaceHolder PH) {
  if (PH == PlaceHolder::Handle)
    return getResourceHandleExpr();
  if (PH == PlaceHolder::CounterHandle)
    return getResourceCounterHandleExpr();

  if (PH == PlaceHolder::LastStmt) {
    assert(!StmtsList.empty() && "no statements in the list");
    Stmt *LastStmt = StmtsList.pop_back_val();
    assert(isa<ValueStmt>(LastStmt) && "last statement does not have a value");
    return cast<ValueStmt>(LastStmt)->getExprStmt();
  }

  // All other placeholders are parameters (_N), and can be loaded as an
  // LValue. It needs to be an LValue if the result expression will be used as
  // the actual parameter for an out parameter. The dimension builtins are an
  // example where this happens.
  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  ParmVarDecl *ParamDecl = Method->getParamDecl(static_cast<unsigned>(PH));
  return DeclRefExpr::Create(
      AST, NestedNameSpecifierLoc(), SourceLocation(), ParamDecl, false,
      DeclarationNameInfo(ParamDecl->getDeclName(), SourceLocation()),
      ParamDecl->getType().getNonReferenceType(), VK_LValue);
}

Expr *BuiltinTypeMethodBuilder::convertPlaceholder(LocalVar &Var) {
  VarDecl *VD = Var.Decl;
  assert(VD && "local variable is not declared");
  return DeclRefExpr::Create(
      VD->getASTContext(), NestedNameSpecifierLoc(), SourceLocation(), VD,
      false, DeclarationNameInfo(VD->getDeclName(), SourceLocation()),
      VD->getType(), VK_LValue);
}

Expr *BuiltinTypeMethodBuilder::convertPlaceholder(QualType Ty) {
  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  QualType PtrTy = AST.getPointerType(Ty);
  // Creates a value-initialized null pointer of type Ty*.
  return new (AST) CXXScalarValueInitExpr(
      PtrTy, AST.getTrivialTypeSourceInfo(PtrTy, SourceLocation()),
      SourceLocation());
}

BuiltinTypeMethodBuilder::BuiltinTypeMethodBuilder(BuiltinTypeDeclBuilder &DB,
                                                   StringRef NameStr,
                                                   QualType ReturnTy,
                                                   bool IsConst, bool IsCtor,
                                                   StorageClass SC)
    : DeclBuilder(DB), ReturnTy(ReturnTy), Method(nullptr), IsConst(IsConst),
      IsCtor(IsCtor), SC(SC) {

  assert((!NameStr.empty() || IsCtor) && "method needs a name");
  assert(((IsCtor && !IsConst) || !IsCtor) && "constructor cannot be const");

  ASTContext &AST = DB.SemaRef.getASTContext();
  if (IsCtor) {
    Name = AST.DeclarationNames.getCXXConstructorName(
        AST.getCanonicalTagType(DB.Record));
  } else {
    const IdentifierInfo &II =
        AST.Idents.get(NameStr, tok::TokenKind::identifier);
    Name = DeclarationName(&II);
  }
}

BuiltinTypeMethodBuilder &
BuiltinTypeMethodBuilder::addParam(StringRef Name, QualType Ty,
                                   HLSLParamModifierAttr::Spelling Modifier) {
  assert(Method == nullptr && "Cannot add param, method already created");
  const IdentifierInfo &II = DeclBuilder.SemaRef.getASTContext().Idents.get(
      Name, tok::TokenKind::identifier);
  Params.emplace_back(II, Ty, Modifier);
  return *this;
}
QualType BuiltinTypeMethodBuilder::addTemplateTypeParam(StringRef Name) {
  assert(Method == nullptr &&
         "Cannot add template param, method already created");
  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  unsigned Position = static_cast<unsigned>(TemplateParamDecls.size());
  auto *Decl = TemplateTypeParmDecl::Create(
      AST, DeclBuilder.Record, SourceLocation(), SourceLocation(),
      /* TemplateDepth */ 0, Position,
      &AST.Idents.get(Name, tok::TokenKind::identifier),
      /* Typename */ true,
      /* ParameterPack */ false,
      /* HasTypeConstraint*/ false);
  TemplateParamDecls.push_back(Decl);

  return QualType(Decl->getTypeForDecl(), 0);
}

void BuiltinTypeMethodBuilder::createDecl() {
  assert(Method == nullptr && "Method or constructor is already created");

  // create function prototype
  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  SmallVector<QualType> ParamTypes;
  SmallVector<FunctionType::ExtParameterInfo> ParamExtInfos(Params.size());
  uint32_t ArgIndex = 0;

  // Create function prototype.
  bool UseParamExtInfo = false;
  for (Param &MP : Params) {
    if (MP.Modifier != HLSLParamModifierAttr::Keyword_in) {
      UseParamExtInfo = true;
      FunctionType::ExtParameterInfo &PI = ParamExtInfos[ArgIndex];
      ParamExtInfos[ArgIndex] =
          PI.withABI(convertParamModifierToParamABI(MP.Modifier));
      if (!MP.Ty->isDependentType())
        MP.Ty = getInoutParameterType(AST, MP.Ty);
    }
    ParamTypes.emplace_back(MP.Ty);
    ++ArgIndex;
  }

  FunctionProtoType::ExtProtoInfo ExtInfo;
  if (UseParamExtInfo)
    ExtInfo.ExtParameterInfos = ParamExtInfos.data();
  if (IsConst)
    ExtInfo.TypeQuals.addConst();

  QualType FuncTy = AST.getFunctionType(ReturnTy, ParamTypes, ExtInfo);

  // Create method or constructor declaration.
  auto *TSInfo = AST.getTrivialTypeSourceInfo(FuncTy, SourceLocation());
  DeclarationNameInfo NameInfo = DeclarationNameInfo(Name, SourceLocation());
  if (IsCtor)
    Method = CXXConstructorDecl::Create(
        AST, DeclBuilder.Record, SourceLocation(), NameInfo, FuncTy, TSInfo,
        ExplicitSpecifier(), false, /*IsInline=*/true, false,
        ConstexprSpecKind::Unspecified);
  else
    Method = CXXMethodDecl::Create(
        AST, DeclBuilder.Record, SourceLocation(), NameInfo, FuncTy, TSInfo, SC,
        false, true, ConstexprSpecKind::Unspecified, SourceLocation());

  // Create params & set them to the method/constructor and function prototype.
  SmallVector<ParmVarDecl *> ParmDecls;
  unsigned CurScopeDepth = DeclBuilder.SemaRef.getCurScope()->getDepth();
  auto FnProtoLoc =
      Method->getTypeSourceInfo()->getTypeLoc().getAs<FunctionProtoTypeLoc>();
  for (int I = 0, E = Params.size(); I != E; I++) {
    Param &MP = Params[I];
    ParmVarDecl *Parm = ParmVarDecl::Create(
        AST, Method, SourceLocation(), SourceLocation(), &MP.NameII, MP.Ty,
        AST.getTrivialTypeSourceInfo(MP.Ty, SourceLocation()), SC_None,
        nullptr);
    if (MP.Modifier != HLSLParamModifierAttr::Keyword_in) {
      auto *Mod =
          HLSLParamModifierAttr::Create(AST, SourceRange(), MP.Modifier);
      Parm->addAttr(Mod);
    }
    Parm->setScopeInfo(CurScopeDepth, I);
    ParmDecls.push_back(Parm);
    FnProtoLoc.setParam(I, Parm);
  }
  Method->setParams({ParmDecls});
}

Expr *BuiltinTypeMethodBuilder::getResourceHandleExpr() {
  ensureCompleteDecl();

  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  CXXThisExpr *This = CXXThisExpr::Create(
      AST, SourceLocation(), Method->getFunctionObjectParameterType(), true);
  FieldDecl *HandleField = DeclBuilder.getResourceHandleField();
  return MemberExpr::CreateImplicit(AST, This, false, HandleField,
                                    HandleField->getType(), VK_LValue,
                                    OK_Ordinary);
}

Expr *BuiltinTypeMethodBuilder::getResourceCounterHandleExpr() {
  ensureCompleteDecl();

  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  CXXThisExpr *This = CXXThisExpr::Create(
      AST, SourceLocation(), Method->getFunctionObjectParameterType(), true);
  FieldDecl *HandleField = DeclBuilder.getResourceCounterHandleField();
  return MemberExpr::CreateImplicit(AST, This, false, HandleField,
                                    HandleField->getType(), VK_LValue,
                                    OK_Ordinary);
}

BuiltinTypeMethodBuilder &
BuiltinTypeMethodBuilder::declareLocalVar(LocalVar &Var) {
  ensureCompleteDecl();

  assert(Var.Decl == nullptr && "local variable is already declared");

  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  Var.Decl = VarDecl::Create(
      AST, Method, SourceLocation(), SourceLocation(),
      &AST.Idents.get(Var.Name, tok::TokenKind::identifier), Var.Ty,
      AST.getTrivialTypeSourceInfo(Var.Ty, SourceLocation()), SC_None);
  DeclStmt *DS = new (AST) clang::DeclStmt(DeclGroupRef(Var.Decl),
                                           SourceLocation(), SourceLocation());
  StmtsList.push_back(DS);
  return *this;
}

BuiltinTypeMethodBuilder &BuiltinTypeMethodBuilder::returnThis() {
  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  CXXThisExpr *ThisExpr = CXXThisExpr::Create(
      AST, SourceLocation(), Method->getFunctionObjectParameterType(),
      /*IsImplicit=*/true);
  StmtsList.push_back(ThisExpr);
  return *this;
}

template <typename... Ts>
BuiltinTypeMethodBuilder &
BuiltinTypeMethodBuilder::callBuiltin(StringRef BuiltinName,
                                      QualType ReturnType, Ts &&...ArgSpecs) {
  ensureCompleteDecl();

  std::array<Expr *, sizeof...(ArgSpecs)> Args{
      convertPlaceholder(std::forward<Ts>(ArgSpecs))...};

  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  FunctionDecl *FD = lookupBuiltinFunction(DeclBuilder.SemaRef, BuiltinName);
  DeclRefExpr *DRE = DeclRefExpr::Create(
      AST, NestedNameSpecifierLoc(), SourceLocation(), FD, false,
      FD->getNameInfo(), AST.BuiltinFnTy, VK_PRValue);

  ExprResult Call = DeclBuilder.SemaRef.BuildCallExpr(
      /*Scope=*/nullptr, DRE, SourceLocation(),
      MultiExprArg(Args.data(), Args.size()), SourceLocation());
  assert(!Call.isInvalid() && "Call to builtin cannot fail!");
  Expr *E = Call.get();

  if (!ReturnType.isNull() &&
      !AST.hasSameUnqualifiedType(ReturnType, E->getType())) {
    ExprResult CastResult = DeclBuilder.SemaRef.BuildCStyleCastExpr(
        SourceLocation(), AST.getTrivialTypeSourceInfo(ReturnType),
        SourceLocation(), E);
    assert(!CastResult.isInvalid() && "Cast cannot fail!");
    E = CastResult.get();
  }

  StmtsList.push_back(E);
  return *this;
}

template <typename TLHS, typename TRHS>
BuiltinTypeMethodBuilder &BuiltinTypeMethodBuilder::assign(TLHS LHS, TRHS RHS) {
  Expr *LHSExpr = convertPlaceholder(LHS);
  Expr *RHSExpr = convertPlaceholder(RHS);
  Stmt *AssignStmt = BinaryOperator::Create(
      DeclBuilder.SemaRef.getASTContext(), LHSExpr, RHSExpr, BO_Assign,
      LHSExpr->getType(), ExprValueKind::VK_PRValue,
      ExprObjectKind::OK_Ordinary, SourceLocation(), FPOptionsOverride());
  StmtsList.push_back(AssignStmt);
  return *this;
}

template <typename T>
BuiltinTypeMethodBuilder &BuiltinTypeMethodBuilder::dereference(T Ptr) {
  Expr *PtrExpr = convertPlaceholder(Ptr);
  Expr *Deref =
      UnaryOperator::Create(DeclBuilder.SemaRef.getASTContext(), PtrExpr,
                            UO_Deref, PtrExpr->getType()->getPointeeType(),
                            VK_LValue, OK_Ordinary, SourceLocation(),
                            /*CanOverflow=*/false, FPOptionsOverride());
  StmtsList.push_back(Deref);
  return *this;
}

template <typename T>
BuiltinTypeMethodBuilder &
BuiltinTypeMethodBuilder::accessHandleFieldOnResource(T ResourceRecord) {
  ensureCompleteDecl();

  Expr *ResourceExpr = convertPlaceholder(ResourceRecord);
  auto *ResourceTypeDecl = ResourceExpr->getType()->getAsCXXRecordDecl();

  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  FieldDecl *HandleField = nullptr;

  if (ResourceTypeDecl == DeclBuilder.Record)
    HandleField = DeclBuilder.getResourceHandleField();
  else {
    IdentifierInfo &II = AST.Idents.get("__handle");
    for (auto *Decl : ResourceTypeDecl->lookup(&II)) {
      if ((HandleField = dyn_cast<FieldDecl>(Decl)))
        break;
    }
    assert(HandleField && "Resource handle field not found");
  }

  MemberExpr *HandleExpr = MemberExpr::CreateImplicit(
      AST, ResourceExpr, false, HandleField, HandleField->getType(), VK_LValue,
      OK_Ordinary);
  StmtsList.push_back(HandleExpr);
  return *this;
}

template <typename ResourceT, typename ValueT>
BuiltinTypeMethodBuilder &
BuiltinTypeMethodBuilder::setHandleFieldOnResource(ResourceT ResourceRecord,
                                                   ValueT HandleValue) {
  return setFieldOnResource(ResourceRecord, HandleValue,
                            DeclBuilder.getResourceHandleField());
}

template <typename ResourceT, typename ValueT>
BuiltinTypeMethodBuilder &
BuiltinTypeMethodBuilder::setCounterHandleFieldOnResource(
    ResourceT ResourceRecord, ValueT HandleValue) {
  return setFieldOnResource(ResourceRecord, HandleValue,
                            DeclBuilder.getResourceCounterHandleField());
}

template <typename ResourceT, typename ValueT>
BuiltinTypeMethodBuilder &BuiltinTypeMethodBuilder::setFieldOnResource(
    ResourceT ResourceRecord, ValueT HandleValue, FieldDecl *HandleField) {
  ensureCompleteDecl();

  Expr *ResourceExpr = convertPlaceholder(ResourceRecord);
  assert(ResourceExpr->getType()->getAsCXXRecordDecl() ==
             HandleField->getParent() &&
         "Getting the field from the wrong resource type.");

  Expr *HandleValueExpr = convertPlaceholder(HandleValue);

  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  MemberExpr *HandleMemberExpr = MemberExpr::CreateImplicit(
      AST, ResourceExpr, false, HandleField, HandleField->getType(), VK_LValue,
      OK_Ordinary);
  Stmt *AssignStmt = BinaryOperator::Create(
      DeclBuilder.SemaRef.getASTContext(), HandleMemberExpr, HandleValueExpr,
      BO_Assign, HandleMemberExpr->getType(), ExprValueKind::VK_PRValue,
      ExprObjectKind::OK_Ordinary, SourceLocation(), FPOptionsOverride());
  StmtsList.push_back(AssignStmt);
  return *this;
}

template <typename T>
BuiltinTypeMethodBuilder &
BuiltinTypeMethodBuilder::accessCounterHandleFieldOnResource(T ResourceRecord) {
  ensureCompleteDecl();

  Expr *ResourceExpr = convertPlaceholder(ResourceRecord);
  assert(ResourceExpr->getType()->getAsCXXRecordDecl() == DeclBuilder.Record &&
         "Getting the field from the wrong resource type.");

  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
  FieldDecl *HandleField = DeclBuilder.getResourceCounterHandleField();
  MemberExpr *HandleExpr = MemberExpr::CreateImplicit(
      AST, ResourceExpr, false, HandleField, HandleField->getType(), VK_LValue,
      OK_Ordinary);
  StmtsList.push_back(HandleExpr);
  return *this;
}

template <typename T>
BuiltinTypeMethodBuilder &BuiltinTypeMethodBuilder::returnValue(T ReturnValue) {
  ensureCompleteDecl();

  Expr *ReturnValueExpr = convertPlaceholder(ReturnValue);
  ASTContext &AST = DeclBuilder.SemaRef.getASTContext();

  QualType Ty = ReturnValueExpr->getType();
  if (Ty->isRecordType()) {
    // For record types, create a call to copy constructor to ensure proper copy
    // semantics.
    auto *ICE =
        ImplicitCastExpr::Create(AST, Ty.withConst(), CK_NoOp, ReturnValueExpr,
                                 nullptr, VK_XValue, FPOptionsOverride());
    CXXConstructorDecl *CD = lookupCopyConstructor(Ty);
    assert(CD && "no copy constructor found");
    ReturnValueExpr = CXXConstructExpr::Create(
        AST, Ty, SourceLocation(), CD, /*Elidable=*/false, {ICE},
        /*HadMultipleCandidates=*/false, /*ListInitialization=*/false,
        /*StdInitListInitialization=*/false,
        /*ZeroInitListInitialization=*/false, CXXConstructionKind::Complete,
        SourceRange());
  }
  StmtsList.push_back(
      ReturnStmt::Create(AST, SourceLocation(), ReturnValueExpr, nullptr));
  return *this;
}

BuiltinTypeDeclBuilder &BuiltinTypeMethodBuilder::finalize() {
  assert(!DeclBuilder.Record->isCompleteDefinition() &&
         "record is already complete");

  ensureCompleteDecl();

  if (!Method->hasBody()) {
    ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
    assert((ReturnTy == AST.VoidTy || !StmtsList.empty()) &&
           "nothing to return from non-void method");
    if (ReturnTy != AST.VoidTy) {
      if (Expr *LastExpr = dyn_cast<Expr>(StmtsList.back())) {
        assert(AST.hasSameUnqualifiedType(LastExpr->getType(),
                                          ReturnTy.getNonReferenceType()) &&
               "Return type of the last statement must match the return type "
               "of the method");
        if (!isa<ReturnStmt>(LastExpr)) {
          StmtsList.pop_back();
          StmtsList.push_back(
              ReturnStmt::Create(AST, SourceLocation(), LastExpr, nullptr));
        }
      }
    }

    Method->setBody(CompoundStmt::Create(AST, StmtsList, FPOptionsOverride(),
                                         SourceLocation(), SourceLocation()));
    Method->setLexicalDeclContext(DeclBuilder.Record);
    Method->setAccess(AS_public);
    Method->addAttr(AlwaysInlineAttr::CreateImplicit(
        AST, SourceRange(), AlwaysInlineAttr::CXX11_clang_always_inline));
    if (!TemplateParamDecls.empty()) {
      TemplateParams = TemplateParameterList::Create(
          AST, SourceLocation(), SourceLocation(), TemplateParamDecls,
          SourceLocation(), nullptr);

      auto *FuncTemplate = FunctionTemplateDecl::Create(AST, DeclBuilder.Record,
                                                        SourceLocation(), Name,
                                                        TemplateParams, Method);
      FuncTemplate->setAccess(AS_public);
      FuncTemplate->setLexicalDeclContext(DeclBuilder.Record);
      FuncTemplate->setImplicit(true);
      Method->setDescribedFunctionTemplate(FuncTemplate);
      DeclBuilder.Record->addDecl(FuncTemplate);
    } else {
      DeclBuilder.Record->addDecl(Method);
    }
  }
  return DeclBuilder;
}

BuiltinTypeDeclBuilder::BuiltinTypeDeclBuilder(Sema &SemaRef, CXXRecordDecl *R)
    : SemaRef(SemaRef), Record(R) {
  Record->startDefinition();
  Template = Record->getDescribedClassTemplate();
}

BuiltinTypeDeclBuilder::BuiltinTypeDeclBuilder(Sema &SemaRef,
                                               NamespaceDecl *Namespace,
                                               StringRef Name)
    : SemaRef(SemaRef), HLSLNamespace(Namespace) {
  ASTContext &AST = SemaRef.getASTContext();
  IdentifierInfo &II = AST.Idents.get(Name, tok::TokenKind::identifier);

  LookupResult Result(SemaRef, &II, SourceLocation(), Sema::LookupTagName);
  CXXRecordDecl *PrevDecl = nullptr;
  if (SemaRef.LookupQualifiedName(Result, HLSLNamespace)) {
    // Declaration already exists (from precompiled headers)
    NamedDecl *Found = Result.getFoundDecl();
    if (auto *TD = dyn_cast<ClassTemplateDecl>(Found)) {
      PrevDecl = TD->getTemplatedDecl();
      PrevTemplate = TD;
    } else
      PrevDecl = dyn_cast<CXXRecordDecl>(Found);
    assert(PrevDecl && "Unexpected lookup result type.");
  }

  if (PrevDecl && PrevDecl->isCompleteDefinition()) {
    Record = PrevDecl;
    Template = PrevTemplate;
    return;
  }

  Record =
      CXXRecordDecl::Create(AST, TagDecl::TagKind::Class, HLSLNamespace,
                            SourceLocation(), SourceLocation(), &II, PrevDecl);
  Record->setImplicit(true);
  Record->setLexicalDeclContext(HLSLNamespace);
  Record->setHasExternalLexicalStorage();

  // Don't let anyone derive from built-in types.
  Record->addAttr(
      FinalAttr::CreateImplicit(AST, SourceRange(), FinalAttr::Keyword_final));
}

BuiltinTypeDeclBuilder::~BuiltinTypeDeclBuilder() {
  if (HLSLNamespace && !Template && Record->getDeclContext() == HLSLNamespace)
    HLSLNamespace->addDecl(Record);
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addMemberVariable(StringRef Name, QualType Type,
                                          llvm::ArrayRef<Attr *> Attrs,
                                          AccessSpecifier Access) {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  assert(Record->isBeingDefined() &&
         "Definition must be started before adding members!");
  ASTContext &AST = Record->getASTContext();

  IdentifierInfo &II = AST.Idents.get(Name, tok::TokenKind::identifier);
  TypeSourceInfo *MemTySource =
      AST.getTrivialTypeSourceInfo(Type, SourceLocation());
  auto *Field = FieldDecl::Create(
      AST, Record, SourceLocation(), SourceLocation(), &II, Type, MemTySource,
      nullptr, false, InClassInitStyle::ICIS_NoInit);
  Field->setAccess(Access);
  Field->setImplicit(true);
  for (Attr *A : Attrs) {
    if (A)
      Field->addAttr(A);
  }

  Record->addDecl(Field);
  Fields[Name] = Field;
  return *this;
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addBufferHandles(ResourceClass RC, bool IsROV,
                                         bool RawBuffer, bool HasCounter,
                                         AccessSpecifier Access) {
  addHandleMember(RC, ResourceDimension::Unknown, IsROV, RawBuffer, Access);
  if (HasCounter)
    addCounterHandleMember(RC, IsROV, RawBuffer, Access);
  return *this;
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addTextureHandle(ResourceClass RC, bool IsROV,
                                         ResourceDimension RD,
                                         AccessSpecifier Access) {
  addHandleMember(RC, RD, IsROV, /*RawBuffer=*/false, Access);
  return *this;
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addSamplerHandle() {
  addHandleMember(ResourceClass::Sampler, ResourceDimension::Unknown,
                  /*IsROV=*/false, /*RawBuffer=*/false);
  return *this;
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addHandleMember(ResourceClass RC, ResourceDimension RD,
                                        bool IsROV, bool RawBuffer,
                                        AccessSpecifier Access) {
  return addResourceMember("__handle", RC, RD, IsROV, RawBuffer,
                           /*IsCounter=*/false, Access);
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addCounterHandleMember(
    ResourceClass RC, bool IsROV, bool RawBuffer, AccessSpecifier Access) {
  return addResourceMember("__counter_handle", RC, ResourceDimension::Unknown,
                           IsROV, RawBuffer,
                           /*IsCounter=*/true, Access);
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addResourceMember(
    StringRef MemberName, ResourceClass RC, ResourceDimension RD, bool IsROV,
    bool RawBuffer, bool IsCounter, AccessSpecifier Access) {
  assert(!Record->isCompleteDefinition() && "record is already complete");

  ASTContext &Ctx = SemaRef.getASTContext();
  TypeSourceInfo *ElementTypeInfo =
      Ctx.getTrivialTypeSourceInfo(getHandleElementType(), SourceLocation());

  // add handle member with resource type attributes
  QualType AttributedResTy = QualType();
  SmallVector<const Attr *> Attrs = {
      HLSLResourceClassAttr::CreateImplicit(Ctx, RC),
      IsROV ? HLSLROVAttr::CreateImplicit(Ctx) : nullptr,
      RawBuffer ? HLSLRawBufferAttr::CreateImplicit(Ctx) : nullptr,
      RD != ResourceDimension::Unknown
          ? HLSLResourceDimensionAttr::CreateImplicit(Ctx, RD)
          : nullptr,
      ElementTypeInfo && RC != ResourceClass::Sampler
          ? HLSLContainedTypeAttr::CreateImplicit(Ctx, ElementTypeInfo)
          : nullptr};
  if (IsCounter)
    Attrs.push_back(HLSLIsCounterAttr::CreateImplicit(Ctx));

  if (CreateHLSLAttributedResourceType(SemaRef, Ctx.HLSLResourceTy, Attrs,
                                       AttributedResTy))
    addMemberVariable(MemberName, AttributedResTy, {}, Access);
  return *this;
}

// Adds default constructor to the resource class:
// Resource::Resource()
BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addDefaultHandleConstructor() {
  assert(!Record->isCompleteDefinition() && "record is already complete");

  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  QualType HandleType = getResourceHandleField()->getType();
  return BuiltinTypeMethodBuilder(*this, "", SemaRef.getASTContext().VoidTy,
                                  false, true)
      .callBuiltin("__builtin_hlsl_resource_uninitializedhandle", HandleType,
                   PH::Handle)
      .assign(PH::Handle, PH::LastStmt)
      .finalize();
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addStaticInitializationFunctions(bool HasCounter) {
  if (HasCounter) {
    addCreateFromBindingWithImplicitCounter();
    addCreateFromImplicitBindingWithImplicitCounter();
  } else {
    addCreateFromBinding();
    addCreateFromImplicitBinding();
  }
  return *this;
}

// Adds static method that initializes resource from binding:
//
// static Resource<T> __createFromBinding(unsigned registerNo,
//                                       unsigned spaceNo, int range,
//                                       unsigned index, const char *name) {
//   Resource<T> tmp;
//   tmp.__handle = __builtin_hlsl_resource_handlefrombinding(
//                                       tmp.__handle, registerNo, spaceNo,
//                                       range, index, name);
//   return tmp;
// }
BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addCreateFromBinding() {
  assert(!Record->isCompleteDefinition() && "record is already complete");

  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  ASTContext &AST = SemaRef.getASTContext();
  QualType HandleType = getResourceHandleField()->getType();
  QualType RecordType = AST.getTypeDeclType(cast<TypeDecl>(Record));
  BuiltinTypeMethodBuilder::LocalVar TmpVar("tmp", RecordType);

  return BuiltinTypeMethodBuilder(*this, "__createFromBinding", RecordType,
                                  false, false, SC_Static)
      .addParam("registerNo", AST.UnsignedIntTy)
      .addParam("spaceNo", AST.UnsignedIntTy)
      .addParam("range", AST.IntTy)
      .addParam("index", AST.UnsignedIntTy)
      .addParam("name", AST.getPointerType(AST.CharTy.withConst()))
      .declareLocalVar(TmpVar)
      .accessHandleFieldOnResource(TmpVar)
      .callBuiltin("__builtin_hlsl_resource_handlefrombinding", HandleType,
                   PH::LastStmt, PH::_0, PH::_1, PH::_2, PH::_3, PH::_4)
      .setHandleFieldOnResource(TmpVar, PH::LastStmt)
      .returnValue(TmpVar)
      .finalize();
}

// Adds static method that initializes resource from binding:
//
// static Resource<T> __createFromImplicitBinding(unsigned orderId,
//                                                unsigned spaceNo, int range,
//                                                unsigned index,
//                                                const char *name) {
//   Resource<T> tmp;
//   tmp.__handle = __builtin_hlsl_resource_handlefromimplicitbinding(
//                                                tmp.__handle, spaceNo,
//                                                range, index, orderId, name);
//   return tmp;
// }
BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addCreateFromImplicitBinding() {
  assert(!Record->isCompleteDefinition() && "record is already complete");

  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  ASTContext &AST = SemaRef.getASTContext();
  QualType HandleType = getResourceHandleField()->getType();
  QualType RecordType = AST.getTypeDeclType(cast<TypeDecl>(Record));
  BuiltinTypeMethodBuilder::LocalVar TmpVar("tmp", RecordType);

  return BuiltinTypeMethodBuilder(*this, "__createFromImplicitBinding",
                                  RecordType, false, false, SC_Static)
      .addParam("orderId", AST.UnsignedIntTy)
      .addParam("spaceNo", AST.UnsignedIntTy)
      .addParam("range", AST.IntTy)
      .addParam("index", AST.UnsignedIntTy)
      .addParam("name", AST.getPointerType(AST.CharTy.withConst()))
      .declareLocalVar(TmpVar)
      .accessHandleFieldOnResource(TmpVar)
      .callBuiltin("__builtin_hlsl_resource_handlefromimplicitbinding",
                   HandleType, PH::LastStmt, PH::_0, PH::_1, PH::_2, PH::_3,
                   PH::_4)
      .setHandleFieldOnResource(TmpVar, PH::LastStmt)
      .returnValue(TmpVar)
      .finalize();
}

// Adds static method that initializes resource from binding:
//
// static Resource<T>
// __createFromBindingWithImplicitCounter(unsigned registerNo,
//                                        unsigned spaceNo, int range,
//                                        unsigned index, const char *name,
//                                        unsigned counterOrderId) {
//   Resource<T> tmp;
//   tmp.__handle = __builtin_hlsl_resource_handlefrombinding(
//       tmp.__handle, registerNo, spaceNo, range, index, name);
//   tmp.__counter_handle =
//       __builtin_hlsl_resource_counterhandlefromimplicitbinding(
//           tmp.__handle, counterOrderId, spaceNo);
//   return tmp;
// }
BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addCreateFromBindingWithImplicitCounter() {
  assert(!Record->isCompleteDefinition() && "record is already complete");

  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  ASTContext &AST = SemaRef.getASTContext();
  QualType HandleType = getResourceHandleField()->getType();
  QualType CounterHandleType = getResourceCounterHandleField()->getType();
  QualType RecordType = AST.getTypeDeclType(cast<TypeDecl>(Record));
  BuiltinTypeMethodBuilder::LocalVar TmpVar("tmp", RecordType);

  return BuiltinTypeMethodBuilder(*this,
                                  "__createFromBindingWithImplicitCounter",
                                  RecordType, false, false, SC_Static)
      .addParam("registerNo", AST.UnsignedIntTy)
      .addParam("spaceNo", AST.UnsignedIntTy)
      .addParam("range", AST.IntTy)
      .addParam("index", AST.UnsignedIntTy)
      .addParam("name", AST.getPointerType(AST.CharTy.withConst()))
      .addParam("counterOrderId", AST.UnsignedIntTy)
      .declareLocalVar(TmpVar)
      .accessHandleFieldOnResource(TmpVar)
      .callBuiltin("__builtin_hlsl_resource_handlefrombinding", HandleType,
                   PH::LastStmt, PH::_0, PH::_1, PH::_2, PH::_3, PH::_4)
      .setHandleFieldOnResource(TmpVar, PH::LastStmt)
      .accessHandleFieldOnResource(TmpVar)
      .callBuiltin("__builtin_hlsl_resource_counterhandlefromimplicitbinding",
                   CounterHandleType, PH::LastStmt, PH::_5, PH::_1)
      .setCounterHandleFieldOnResource(TmpVar, PH::LastStmt)
      .returnValue(TmpVar)
      .finalize();
}

// Adds static method that initializes resource from binding:
//
// static Resource<T>
// __createFromImplicitBindingWithImplicitCounter(unsigned orderId,
//                                                unsigned spaceNo, int range,
//                                                unsigned index,
//                                                const char *name,
//                                                unsigned counterOrderId) {
//   Resource<T> tmp;
//   tmp.__handle = __builtin_hlsl_resource_handlefromimplicitbinding(
//       tmp.__handle, orderId, spaceNo, range, index, name);
//   tmp.__counter_handle =
//       __builtin_hlsl_resource_counterhandlefromimplicitbinding(
//           tmp.__handle, counterOrderId, spaceNo);
//   return tmp;
// }
BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addCreateFromImplicitBindingWithImplicitCounter() {
  assert(!Record->isCompleteDefinition() && "record is already complete");

  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  ASTContext &AST = SemaRef.getASTContext();
  QualType HandleType = getResourceHandleField()->getType();
  QualType CounterHandleType = getResourceCounterHandleField()->getType();
  QualType RecordType = AST.getTypeDeclType(cast<TypeDecl>(Record));
  BuiltinTypeMethodBuilder::LocalVar TmpVar("tmp", RecordType);

  return BuiltinTypeMethodBuilder(
             *this, "__createFromImplicitBindingWithImplicitCounter",
             RecordType, false, false, SC_Static)
      .addParam("orderId", AST.UnsignedIntTy)
      .addParam("spaceNo", AST.UnsignedIntTy)
      .addParam("range", AST.IntTy)
      .addParam("index", AST.UnsignedIntTy)
      .addParam("name", AST.getPointerType(AST.CharTy.withConst()))
      .addParam("counterOrderId", AST.UnsignedIntTy)
      .declareLocalVar(TmpVar)
      .accessHandleFieldOnResource(TmpVar)
      .callBuiltin("__builtin_hlsl_resource_handlefromimplicitbinding",
                   HandleType, PH::LastStmt, PH::_0, PH::_1, PH::_2, PH::_3,
                   PH::_4)
      .setHandleFieldOnResource(TmpVar, PH::LastStmt)
      .accessHandleFieldOnResource(TmpVar)
      .callBuiltin("__builtin_hlsl_resource_counterhandlefromimplicitbinding",
                   CounterHandleType, PH::LastStmt, PH::_5, PH::_1)
      .setCounterHandleFieldOnResource(TmpVar, PH::LastStmt)
      .returnValue(TmpVar)
      .finalize();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addCopyConstructor() {
  assert(!Record->isCompleteDefinition() && "record is already complete");

  ASTContext &AST = SemaRef.getASTContext();
  QualType RecordType = AST.getCanonicalTagType(Record);
  QualType ConstRecordType = RecordType.withConst();
  QualType ConstRecordRefType = AST.getLValueReferenceType(ConstRecordType);

  using PH = BuiltinTypeMethodBuilder::PlaceHolder;

  BuiltinTypeMethodBuilder MMB(*this, /*Name=*/"", AST.VoidTy,
                               /*IsConst=*/false, /*IsCtor=*/true);
  MMB.addParam("other", ConstRecordRefType)
      .accessHandleFieldOnResource(PH::_0)
      .assign(PH::Handle, PH::LastStmt);

  if (getResourceCounterHandleField())
    MMB.accessCounterHandleFieldOnResource(PH::_0).assign(PH::CounterHandle,
                                                          PH::LastStmt);

  return MMB.finalize();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addCopyAssignmentOperator() {
  assert(!Record->isCompleteDefinition() && "record is already complete");

  ASTContext &AST = SemaRef.getASTContext();
  QualType RecordType = AST.getCanonicalTagType(Record);
  QualType ConstRecordType = RecordType.withConst();
  QualType ConstRecordRefType = AST.getLValueReferenceType(ConstRecordType);
  QualType RecordRefType = AST.getLValueReferenceType(RecordType);

  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  DeclarationName Name = AST.DeclarationNames.getCXXOperatorName(OO_Equal);
  BuiltinTypeMethodBuilder MMB(*this, Name, RecordRefType);
  MMB.addParam("other", ConstRecordRefType)
      .accessHandleFieldOnResource(PH::_0)
      .assign(PH::Handle, PH::LastStmt);

  if (getResourceCounterHandleField())
    MMB.accessCounterHandleFieldOnResource(PH::_0).assign(PH::CounterHandle,
                                                          PH::LastStmt);

  return MMB.returnThis().finalize();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addArraySubscriptOperators() {
  ASTContext &AST = Record->getASTContext();
  DeclarationName Subscript =
      AST.DeclarationNames.getCXXOperatorName(OO_Subscript);

  addHandleAccessFunction(Subscript, /*IsConst=*/true, /*IsRef=*/true);
  if (getResourceAttrs().ResourceClass == llvm::dxil::ResourceClass::UAV)
    addHandleAccessFunction(Subscript, /*IsConst=*/false, /*IsRef=*/true);

  return *this;
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addLoadMethods() {
  assert(!Record->isCompleteDefinition() && "record is already complete");

  ASTContext &AST = Record->getASTContext();
  IdentifierInfo &II = AST.Idents.get("Load", tok::TokenKind::identifier);
  DeclarationName Load(&II);
  // TODO: We also need versions with status for CheckAccessFullyMapped.
  addHandleAccessFunction(Load, /*IsConst=*/false, /*IsRef=*/false);
  addLoadWithStatusFunction(Load, /*IsConst=*/false);

  return *this;
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addByteAddressBufferLoadMethods() {
  assert(!Record->isCompleteDefinition() && "record is already complete");

  ASTContext &AST = SemaRef.getASTContext();

  auto AddLoads = [&](StringRef MethodName, QualType ReturnType) {
    IdentifierInfo &II = AST.Idents.get(MethodName, tok::TokenKind::identifier);
    DeclarationName Load(&II);

    addHandleAccessFunction(Load, /*IsConst=*/false, /*IsRef=*/false,
                            ReturnType);
    addLoadWithStatusFunction(Load, /*IsConst=*/false, ReturnType);
  };

  AddLoads("Load", AST.UnsignedIntTy);
  AddLoads("Load2", AST.getExtVectorType(AST.UnsignedIntTy, 2));
  AddLoads("Load3", AST.getExtVectorType(AST.UnsignedIntTy, 3));
  AddLoads("Load4", AST.getExtVectorType(AST.UnsignedIntTy, 4));
  AddLoads("Load", AST.DependentTy); // Templated version
  return *this;
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addByteAddressBufferStoreMethods() {
  assert(!Record->isCompleteDefinition() && "record is already complete");

  ASTContext &AST = SemaRef.getASTContext();

  auto AddStore = [&](StringRef MethodName, QualType ValueType) {
    IdentifierInfo &II = AST.Idents.get(MethodName, tok::TokenKind::identifier);
    DeclarationName Store(&II);

    addStoreFunction(Store, /*IsConst=*/false, ValueType);
  };

  AddStore("Store", AST.UnsignedIntTy);
  AddStore("Store2", AST.getExtVectorType(AST.UnsignedIntTy, 2));
  AddStore("Store3", AST.getExtVectorType(AST.UnsignedIntTy, 3));
  AddStore("Store4", AST.getExtVectorType(AST.UnsignedIntTy, 4));
  AddStore("Store", AST.DependentTy); // Templated version

  return *this;
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addSampleMethods(ResourceDimension Dim) {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  ASTContext &AST = Record->getASTContext();
  QualType ReturnType = getFirstTemplateTypeParam();
  QualType SamplerStateType =
      lookupBuiltinType(SemaRef, "SamplerState", Record->getDeclContext());
  uint32_t VecSize = getResourceDimensions(Dim);
  QualType FloatTy = AST.FloatTy;
  QualType Float2Ty = AST.getExtVectorType(FloatTy, VecSize);
  QualType IntTy = AST.IntTy;
  QualType Int2Ty = AST.getExtVectorType(IntTy, VecSize);
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;

  // T Sample(SamplerState s, float2 location)
  BuiltinTypeMethodBuilder(*this, "Sample", ReturnType)
      .addParam("Sampler", SamplerStateType)
      .addParam("Location", Float2Ty)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample", ReturnType, PH::Handle,
                   PH::LastStmt, PH::_1)
      .returnValue(PH::LastStmt)
      .finalize();

  // T Sample(SamplerState s, float2 location, int2 offset)
  BuiltinTypeMethodBuilder(*this, "Sample", ReturnType)
      .addParam("Sampler", SamplerStateType)
      .addParam("Location", Float2Ty)
      .addParam("Offset", Int2Ty)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample", ReturnType, PH::Handle,
                   PH::LastStmt, PH::_1, PH::_2)
      .returnValue(PH::LastStmt)
      .finalize();

  // T Sample(SamplerState s, float2 location, int2 offset, float clamp)
  return BuiltinTypeMethodBuilder(*this, "Sample", ReturnType)
      .addParam("Sampler", SamplerStateType)
      .addParam("Location", Float2Ty)
      .addParam("Offset", Int2Ty)
      .addParam("Clamp", FloatTy)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample", ReturnType, PH::Handle,
                   PH::LastStmt, PH::_1, PH::_2, PH::_3)
      .returnValue(PH::LastStmt)
      .finalize();
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addSampleBiasMethods(ResourceDimension Dim) {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  ASTContext &AST = Record->getASTContext();
  QualType ReturnType = getFirstTemplateTypeParam();
  QualType SamplerStateType =
      lookupBuiltinType(SemaRef, "SamplerState", Record->getDeclContext());
  uint32_t VecSize = getResourceDimensions(Dim);
  QualType FloatTy = AST.FloatTy;
  QualType Float2Ty = AST.getExtVectorType(FloatTy, VecSize);
  QualType IntTy = AST.IntTy;
  QualType Int2Ty = AST.getExtVectorType(IntTy, VecSize);
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;

  // T SampleBias(SamplerState s, float2 location, float bias)
  BuiltinTypeMethodBuilder(*this, "SampleBias", ReturnType)
      .addParam("Sampler", SamplerStateType)
      .addParam("Location", Float2Ty)
      .addParam("Bias", FloatTy)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample_bias", ReturnType,
                   PH::Handle, PH::LastStmt, PH::_1, PH::_2)
      .returnValue(PH::LastStmt)
      .finalize();

  // T SampleBias(SamplerState s, float2 location, float bias, int2 offset)
  BuiltinTypeMethodBuilder(*this, "SampleBias", ReturnType)
      .addParam("Sampler", SamplerStateType)
      .addParam("Location", Float2Ty)
      .addParam("Bias", FloatTy)
      .addParam("Offset", Int2Ty)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample_bias", ReturnType,
                   PH::Handle, PH::LastStmt, PH::_1, PH::_2, PH::_3)
      .returnValue(PH::LastStmt)
      .finalize();

  // T SampleBias(SamplerState s, float2 location, float bias, int2 offset,
  // float clamp)
  return BuiltinTypeMethodBuilder(*this, "SampleBias", ReturnType)
      .addParam("Sampler", SamplerStateType)
      .addParam("Location", Float2Ty)
      .addParam("Bias", FloatTy)
      .addParam("Offset", Int2Ty)
      .addParam("Clamp", FloatTy)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample_bias", ReturnType,
                   PH::Handle, PH::LastStmt, PH::_1, PH::_2, PH::_3, PH::_4)
      .returnValue(PH::LastStmt)
      .finalize();
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addSampleGradMethods(ResourceDimension Dim) {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  ASTContext &AST = Record->getASTContext();
  QualType ReturnType = getFirstTemplateTypeParam();
  QualType SamplerStateType =
      lookupBuiltinType(SemaRef, "SamplerState", Record->getDeclContext());
  uint32_t VecSize = getResourceDimensions(Dim);
  QualType FloatTy = AST.FloatTy;
  QualType Float2Ty = AST.getExtVectorType(FloatTy, VecSize);
  QualType IntTy = AST.IntTy;
  QualType Int2Ty = AST.getExtVectorType(IntTy, VecSize);
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;

  // T SampleGrad(SamplerState s, float2 location, float2 ddx, float2 ddy)
  BuiltinTypeMethodBuilder(*this, "SampleGrad", ReturnType)
      .addParam("Sampler", SamplerStateType)
      .addParam("Location", Float2Ty)
      .addParam("DDX", Float2Ty)
      .addParam("DDY", Float2Ty)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample_grad", ReturnType,
                   PH::Handle, PH::LastStmt, PH::_1, PH::_2, PH::_3)
      .returnValue(PH::LastStmt)
      .finalize();

  // T SampleGrad(SamplerState s, float2 location, float2 ddx, float2 ddy,
  // int2 offset)
  BuiltinTypeMethodBuilder(*this, "SampleGrad", ReturnType)
      .addParam("Sampler", SamplerStateType)
      .addParam("Location", Float2Ty)
      .addParam("DDX", Float2Ty)
      .addParam("DDY", Float2Ty)
      .addParam("Offset", Int2Ty)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample_grad", ReturnType,
                   PH::Handle, PH::LastStmt, PH::_1, PH::_2, PH::_3, PH::_4)
      .returnValue(PH::LastStmt)
      .finalize();

  // T SampleGrad(SamplerState s, float2 location, float2 ddx, float2 ddy,
  // int2 offset, float clamp)
  return BuiltinTypeMethodBuilder(*this, "SampleGrad", ReturnType)
      .addParam("Sampler", SamplerStateType)
      .addParam("Location", Float2Ty)
      .addParam("DDX", Float2Ty)
      .addParam("DDY", Float2Ty)
      .addParam("Offset", Int2Ty)
      .addParam("Clamp", FloatTy)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample_grad", ReturnType,
                   PH::Handle, PH::LastStmt, PH::_1, PH::_2, PH::_3, PH::_4,
                   PH::_5)
      .returnValue(PH::LastStmt)
      .finalize();
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addSampleLevelMethods(ResourceDimension Dim) {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  ASTContext &AST = Record->getASTContext();
  QualType ReturnType = getFirstTemplateTypeParam();
  QualType SamplerStateType =
      lookupBuiltinType(SemaRef, "SamplerState", Record->getDeclContext());
  uint32_t VecSize = getResourceDimensions(Dim);
  QualType FloatTy = AST.FloatTy;
  QualType Float2Ty = AST.getExtVectorType(FloatTy, VecSize);
  QualType IntTy = AST.IntTy;
  QualType Int2Ty = AST.getExtVectorType(IntTy, VecSize);
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;

  // T SampleLevel(SamplerState s, float2 location, float lod)
  BuiltinTypeMethodBuilder(*this, "SampleLevel", ReturnType)
      .addParam("Sampler", SamplerStateType)
      .addParam("Location", Float2Ty)
      .addParam("LOD", FloatTy)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample_level", ReturnType,
                   PH::Handle, PH::LastStmt, PH::_1, PH::_2)
      .returnValue(PH::LastStmt)
      .finalize();

  // T SampleLevel(SamplerState s, float2 location, float lod, int2 offset)
  return BuiltinTypeMethodBuilder(*this, "SampleLevel", ReturnType)
      .addParam("Sampler", SamplerStateType)
      .addParam("Location", Float2Ty)
      .addParam("LOD", FloatTy)
      .addParam("Offset", Int2Ty)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample_level", ReturnType,
                   PH::Handle, PH::LastStmt, PH::_1, PH::_2, PH::_3)
      .returnValue(PH::LastStmt)
      .finalize();
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addSampleCmpMethods(ResourceDimension Dim) {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  ASTContext &AST = Record->getASTContext();
  QualType ReturnType = AST.FloatTy;
  QualType SamplerComparisonStateType = lookupBuiltinType(
      SemaRef, "SamplerComparisonState", Record->getDeclContext());
  uint32_t VecSize = getResourceDimensions(Dim);
  QualType FloatTy = AST.FloatTy;
  QualType Float2Ty = AST.getExtVectorType(FloatTy, VecSize);
  QualType IntTy = AST.IntTy;
  QualType Int2Ty = AST.getExtVectorType(IntTy, VecSize);
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;

  // T SampleCmp(SamplerComparisonState s, float2 location, float compare_value)
  BuiltinTypeMethodBuilder(*this, "SampleCmp", ReturnType)
      .addParam("Sampler", SamplerComparisonStateType)
      .addParam("Location", Float2Ty)
      .addParam("CompareValue", FloatTy)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample_cmp", ReturnType, PH::Handle,
                   PH::LastStmt, PH::_1, PH::_2)
      .returnValue(PH::LastStmt)
      .finalize();

  // T SampleCmp(SamplerComparisonState s, float2 location, float compare_value,
  // int2 offset)
  BuiltinTypeMethodBuilder(*this, "SampleCmp", ReturnType)
      .addParam("Sampler", SamplerComparisonStateType)
      .addParam("Location", Float2Ty)
      .addParam("CompareValue", FloatTy)
      .addParam("Offset", Int2Ty)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample_cmp", ReturnType, PH::Handle,
                   PH::LastStmt, PH::_1, PH::_2, PH::_3)
      .returnValue(PH::LastStmt)
      .finalize();

  // T SampleCmp(SamplerComparisonState s, float2 location, float compare_value,
  // int2 offset, float clamp)
  return BuiltinTypeMethodBuilder(*this, "SampleCmp", ReturnType)
      .addParam("Sampler", SamplerComparisonStateType)
      .addParam("Location", Float2Ty)
      .addParam("CompareValue", FloatTy)
      .addParam("Offset", Int2Ty)
      .addParam("Clamp", FloatTy)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample_cmp", ReturnType, PH::Handle,
                   PH::LastStmt, PH::_1, PH::_2, PH::_3, PH::_4)
      .returnValue(PH::LastStmt)
      .finalize();
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addSampleCmpLevelZeroMethods(ResourceDimension Dim) {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  ASTContext &AST = Record->getASTContext();
  QualType ReturnType = AST.FloatTy;
  QualType SamplerComparisonStateType = lookupBuiltinType(
      SemaRef, "SamplerComparisonState", Record->getDeclContext());
  uint32_t VecSize = getResourceDimensions(Dim);
  QualType FloatTy = AST.FloatTy;
  QualType Float2Ty = AST.getExtVectorType(FloatTy, VecSize);
  QualType IntTy = AST.IntTy;
  QualType Int2Ty = AST.getExtVectorType(IntTy, VecSize);
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;

  // T SampleCmpLevelZero(SamplerComparisonState s, float2 location, float
  // compare_value)
  BuiltinTypeMethodBuilder(*this, "SampleCmpLevelZero", ReturnType)
      .addParam("Sampler", SamplerComparisonStateType)
      .addParam("Location", Float2Ty)
      .addParam("CompareValue", FloatTy)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample_cmp_level_zero", ReturnType,
                   PH::Handle, PH::LastStmt, PH::_1, PH::_2)
      .returnValue(PH::LastStmt)
      .finalize();

  // T SampleCmpLevelZero(SamplerComparisonState s, float2 location, float
  // compare_value, int2 offset)
  return BuiltinTypeMethodBuilder(*this, "SampleCmpLevelZero", ReturnType)
      .addParam("Sampler", SamplerComparisonStateType)
      .addParam("Location", Float2Ty)
      .addParam("CompareValue", FloatTy)
      .addParam("Offset", Int2Ty)
      .accessHandleFieldOnResource(PH::_0)
      .callBuiltin("__builtin_hlsl_resource_sample_cmp_level_zero", ReturnType,
                   PH::Handle, PH::LastStmt, PH::_1, PH::_2, PH::_3)
      .returnValue(PH::LastStmt)
      .finalize();
}

FieldDecl *BuiltinTypeDeclBuilder::getResourceHandleField() const {
  auto I = Fields.find("__handle");
  assert(I != Fields.end() &&
         I->second->getType()->isHLSLAttributedResourceType() &&
         "record does not have resource handle field");
  return I->second;
}

FieldDecl *BuiltinTypeDeclBuilder::getResourceCounterHandleField() const {
  auto I = Fields.find("__counter_handle");
  if (I == Fields.end() ||
      !I->second->getType()->isHLSLAttributedResourceType())
    return nullptr;
  return I->second;
}

QualType BuiltinTypeDeclBuilder::getFirstTemplateTypeParam() {
  assert(Template && "record it not a template");
  if (const auto *TTD = dyn_cast<TemplateTypeParmDecl>(
          Template->getTemplateParameters()->getParam(0))) {
    return QualType(TTD->getTypeForDecl(), 0);
  }
  return QualType();
}

QualType BuiltinTypeDeclBuilder::getHandleElementType() {
  if (Template)
    return getFirstTemplateTypeParam();
  // TODO: Should we default to VoidTy? Using `i8` is arguably ambiguous.
  return SemaRef.getASTContext().Char8Ty;
}

HLSLAttributedResourceType::Attributes
BuiltinTypeDeclBuilder::getResourceAttrs() const {
  QualType HandleType = getResourceHandleField()->getType();
  return cast<HLSLAttributedResourceType>(HandleType)->getAttrs();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::completeDefinition() {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  assert(Record->isBeingDefined() &&
         "Definition must be started before completing it.");

  Record->completeDefinition();
  return *this;
}

Expr *BuiltinTypeDeclBuilder::getConstantIntExpr(int value) {
  ASTContext &AST = SemaRef.getASTContext();
  return IntegerLiteral::Create(
      AST, llvm::APInt(AST.getTypeSize(AST.IntTy), value, true), AST.IntTy,
      SourceLocation());
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addSimpleTemplateParams(ArrayRef<StringRef> Names,
                                                ConceptDecl *CD = nullptr) {
  if (Record->isCompleteDefinition()) {
    assert(Template && "existing record it not a template");
    assert(Template->getTemplateParameters()->size() == Names.size() &&
           "template param count mismatch");
    return *this;
  }

  TemplateParameterListBuilder Builder = TemplateParameterListBuilder(*this);
  for (StringRef Name : Names)
    Builder.addTypeParameter(Name);
  return Builder.finalizeTemplateArgs(CD);
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addIncrementCounterMethod() {
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  QualType UnsignedIntTy = SemaRef.getASTContext().UnsignedIntTy;
  return BuiltinTypeMethodBuilder(*this, "IncrementCounter", UnsignedIntTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", UnsignedIntTy,
                   PH::CounterHandle, getConstantIntExpr(1))
      .finalize();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addDecrementCounterMethod() {
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  QualType UnsignedIntTy = SemaRef.getASTContext().UnsignedIntTy;
  return BuiltinTypeMethodBuilder(*this, "DecrementCounter", UnsignedIntTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", UnsignedIntTy,
                   PH::CounterHandle, getConstantIntExpr(-1))
      .finalize();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addLoadWithStatusFunction(
    DeclarationName &Name, bool IsConst, QualType ReturnTy) {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  ASTContext &AST = SemaRef.getASTContext();
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  bool NeedsTypedBuiltin = !ReturnTy.isNull();

  // The empty QualType is a placeholder. The actual return type is set below.
  BuiltinTypeMethodBuilder MMB(*this, Name, QualType(), IsConst);

  if (!NeedsTypedBuiltin)
    ReturnTy = getHandleElementType();
  if (ReturnTy == AST.DependentTy)
    ReturnTy = MMB.addTemplateTypeParam("element_type");
  MMB.ReturnTy = ReturnTy;

  MMB.addParam("Index", AST.UnsignedIntTy)
      .addParam("Status", AST.UnsignedIntTy,
                HLSLParamModifierAttr::Keyword_out);

  if (NeedsTypedBuiltin)
    MMB.callBuiltin("__builtin_hlsl_resource_load_with_status_typed", ReturnTy,
                    PH::Handle, PH::_0, PH::_1, ReturnTy);
  else
    MMB.callBuiltin("__builtin_hlsl_resource_load_with_status", ReturnTy,
                    PH::Handle, PH::_0, PH::_1);

  return MMB.finalize();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addHandleAccessFunction(
    DeclarationName &Name, bool IsConst, bool IsRef, QualType ElemTy) {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  ASTContext &AST = SemaRef.getASTContext();
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  bool NeedsTypedBuiltin = !ElemTy.isNull();

  // The empty QualType is a placeholder. The actual return type is set below.
  BuiltinTypeMethodBuilder MMB(*this, Name, QualType(), IsConst);

  if (!NeedsTypedBuiltin)
    ElemTy = getHandleElementType();
  if (ElemTy == AST.DependentTy)
    ElemTy = MMB.addTemplateTypeParam("element_type");
  QualType AddrSpaceElemTy =
      AST.getAddrSpaceQualType(ElemTy, LangAS::hlsl_device);
  QualType ElemPtrTy = AST.getPointerType(AddrSpaceElemTy);
  QualType ReturnTy;

  if (IsRef) {
    ReturnTy = AddrSpaceElemTy;
    if (IsConst)
      ReturnTy.addConst();
    ReturnTy = AST.getLValueReferenceType(ReturnTy);
  } else {
    ReturnTy = ElemTy;
    if (IsConst)
      ReturnTy.addConst();
  }
  MMB.ReturnTy = ReturnTy;

  MMB.addParam("Index", AST.UnsignedIntTy);

  if (NeedsTypedBuiltin)
    MMB.callBuiltin("__builtin_hlsl_resource_getpointer_typed", ElemPtrTy,
                    PH::Handle, PH::_0, ElemTy);
  else
    MMB.callBuiltin("__builtin_hlsl_resource_getpointer", ElemPtrTy, PH::Handle,
                    PH::_0);

  return MMB.dereference(PH::LastStmt).finalize();
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addStoreFunction(DeclarationName &Name, bool IsConst,
                                         QualType ValueTy) {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  ASTContext &AST = SemaRef.getASTContext();
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;

  BuiltinTypeMethodBuilder MMB(*this, Name, AST.VoidTy, IsConst);

  if (ValueTy == AST.DependentTy)
    ValueTy = MMB.addTemplateTypeParam("element_type");
  QualType AddrSpaceElemTy =
      AST.getAddrSpaceQualType(ValueTy, LangAS::hlsl_device);
  QualType ElemPtrTy = AST.getPointerType(AddrSpaceElemTy);

  return MMB.addParam("Index", AST.UnsignedIntTy)
      .addParam("Value", ValueTy)
      .callBuiltin("__builtin_hlsl_resource_getpointer_typed", ElemPtrTy,
                   PH::Handle, PH::_0, ValueTy)
      .dereference(PH::LastStmt)
      .assign(PH::LastStmt, PH::_1)
      .finalize();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addAppendMethod() {
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  ASTContext &AST = SemaRef.getASTContext();
  QualType ElemTy = getHandleElementType();
  QualType AddrSpaceElemTy =
      AST.getAddrSpaceQualType(ElemTy, LangAS::hlsl_device);
  return BuiltinTypeMethodBuilder(*this, "Append", AST.VoidTy)
      .addParam("value", ElemTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", AST.UnsignedIntTy,
                   PH::CounterHandle, getConstantIntExpr(1))
      .callBuiltin("__builtin_hlsl_resource_getpointer",
                   AST.getPointerType(AddrSpaceElemTy), PH::Handle,
                   PH::LastStmt)
      .dereference(PH::LastStmt)
      .assign(PH::LastStmt, PH::_0)
      .finalize();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addConsumeMethod() {
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  ASTContext &AST = SemaRef.getASTContext();
  QualType ElemTy = getHandleElementType();
  QualType AddrSpaceElemTy =
      AST.getAddrSpaceQualType(ElemTy, LangAS::hlsl_device);
  return BuiltinTypeMethodBuilder(*this, "Consume", ElemTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", AST.UnsignedIntTy,
                   PH::CounterHandle, getConstantIntExpr(-1))
      .callBuiltin("__builtin_hlsl_resource_getpointer",
                   AST.getPointerType(AddrSpaceElemTy), PH::Handle,
                   PH::LastStmt)
      .dereference(PH::LastStmt)
      .finalize();
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addGetDimensionsMethodForBuffer() {
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  ASTContext &AST = SemaRef.getASTContext();
  QualType UIntTy = AST.UnsignedIntTy;

  QualType HandleTy = getResourceHandleField()->getType();
  auto *AttrResTy = cast<HLSLAttributedResourceType>(HandleTy.getTypePtr());

  // Structured buffers except {RW}ByteAddressBuffer have overload
  // GetDimensions(out uint numStructs, out uint stride).
  if (AttrResTy->getAttrs().RawBuffer &&
      AttrResTy->getContainedType() != AST.Char8Ty) {
    return BuiltinTypeMethodBuilder(*this, "GetDimensions", AST.VoidTy)
        .addParam("numStructs", UIntTy, HLSLParamModifierAttr::Keyword_out)
        .addParam("stride", UIntTy, HLSLParamModifierAttr::Keyword_out)
        .callBuiltin("__builtin_hlsl_resource_getdimensions_x", QualType(),
                     PH::Handle, PH::_0)
        .callBuiltin("__builtin_hlsl_resource_getstride", QualType(),
                     PH::Handle, PH::_1)
        .finalize();
  }

  // Typed buffers and {RW}ByteAddressBuffer have overload
  // GetDimensions(out uint dim).
  return BuiltinTypeMethodBuilder(*this, "GetDimensions", AST.VoidTy)
      .addParam("dim", UIntTy, HLSLParamModifierAttr::Keyword_out)
      .callBuiltin("__builtin_hlsl_resource_getdimensions_x", QualType(),
                   PH::Handle, PH::_0)
      .finalize();
}

} // namespace hlsl
} // namespace clang
