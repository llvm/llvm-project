//===--- HLSLExternalSemaSource.cpp - HLSL Sema Source --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/HLSLExternalSemaSource.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaHLSL.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/HLSL/HLSLResource.h"
#include "llvm/Support/ErrorHandling.h"

#include <functional>

using namespace clang;
using namespace llvm::hlsl;

static FunctionDecl *lookupBuiltinFunction(Sema &S, StringRef Name);

namespace {

struct TemplateParameterListBuilder;

class BuiltinTypeDeclBuilder {
  ClassTemplateDecl *Template = nullptr;
  ClassTemplateDecl *PrevTemplate = nullptr;
  NamespaceDecl *HLSLNamespace = nullptr;
  llvm::StringMap<FieldDecl *> Fields;

public:
  Sema &SemaRef;
  CXXRecordDecl *Record = nullptr;
  friend struct TemplateParameterListBuilder;

  BuiltinTypeDeclBuilder(Sema &SemaRef, CXXRecordDecl *R)
      : SemaRef(SemaRef), Record(R) {
    Record->startDefinition();
    Template = Record->getDescribedClassTemplate();
  }

  BuiltinTypeDeclBuilder(Sema &SemaRef, NamespaceDecl *Namespace,
                         StringRef Name)
      : HLSLNamespace(Namespace), SemaRef(SemaRef) {
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

    Record = CXXRecordDecl::Create(AST, TagDecl::TagKind::Class, HLSLNamespace,
                                   SourceLocation(), SourceLocation(), &II,
                                   PrevDecl, true);
    Record->setImplicit(true);
    Record->setLexicalDeclContext(HLSLNamespace);
    Record->setHasExternalLexicalStorage();

    // Don't let anyone derive from built-in types.
    Record->addAttr(FinalAttr::CreateImplicit(AST, SourceRange(),
                                              FinalAttr::Keyword_final));
  }

  ~BuiltinTypeDeclBuilder() {
    if (HLSLNamespace && !Template && Record->getDeclContext() == HLSLNamespace)
      HLSLNamespace->addDecl(Record);
  }

  CXXRecordDecl *finalizeForwardDeclaration() {
    // Force the QualType to be generated for the record declaration. In most
    // cases this will happen naturally when something uses the type the
    // QualType gets lazily created. Unfortunately, with our injected types if a
    // type isn't used in a translation unit the QualType may not get
    // automatically generated before a PCH is generated. To resolve this we
    // just force that the QualType is generated after we create a forward
    // declaration.
    (void)Record->getASTContext().getRecordType(Record);
    return Record;
  }

  BuiltinTypeDeclBuilder &
  addMemberVariable(StringRef Name, QualType Type, llvm::ArrayRef<Attr *> Attrs,
                    AccessSpecifier Access = AccessSpecifier::AS_private) {
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
  addHandleMember(ResourceClass RC, ResourceKind RK, bool IsROV, bool RawBuffer,
                  AccessSpecifier Access = AccessSpecifier::AS_private) {
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
        ElementTypeInfo
            ? HLSLContainedTypeAttr::CreateImplicit(Ctx, ElementTypeInfo)
            : nullptr};
    Attr *ResourceAttr = HLSLResourceAttr::CreateImplicit(Ctx, RK);
    if (CreateHLSLAttributedResourceType(SemaRef, Ctx.HLSLResourceTy, Attrs,
                                         AttributedResTy))
      addMemberVariable("__handle", AttributedResTy, {ResourceAttr}, Access);
    return *this;
  }

  BuiltinTypeDeclBuilder &addDefaultHandleConstructor() {
    if (Record->isCompleteDefinition())
      return *this;
    ASTContext &AST = Record->getASTContext();

    QualType ConstructorType =
        AST.getFunctionType(AST.VoidTy, {}, FunctionProtoType::ExtProtoInfo());

    CanQualType CanTy = Record->getTypeForDecl()->getCanonicalTypeUnqualified();
    DeclarationName Name = AST.DeclarationNames.getCXXConstructorName(CanTy);
    CXXConstructorDecl *Constructor = CXXConstructorDecl::Create(
        AST, Record, SourceLocation(),
        DeclarationNameInfo(Name, SourceLocation()), ConstructorType,
        AST.getTrivialTypeSourceInfo(ConstructorType, SourceLocation()),
        ExplicitSpecifier(), false, true, false,
        ConstexprSpecKind::Unspecified);

    Constructor->setBody(CompoundStmt::Create(
        AST, {}, FPOptionsOverride(), SourceLocation(), SourceLocation()));
    Constructor->setAccess(AccessSpecifier::AS_public);
    Record->addDecl(Constructor);
    return *this;
  }

  BuiltinTypeDeclBuilder &addArraySubscriptOperators() {
    ASTContext &AST = Record->getASTContext();
    DeclarationName Subscript =
        AST.DeclarationNames.getCXXOperatorName(OO_Subscript);

    addHandleAccessFunction(Subscript, /*IsConst=*/true, /*IsRef=*/true);
    addHandleAccessFunction(Subscript, /*IsConst=*/false, /*IsRef=*/true);
    return *this;
  }

  BuiltinTypeDeclBuilder &addLoadMethods() {
    if (Record->isCompleteDefinition())
      return *this;

    ASTContext &AST = Record->getASTContext();
    IdentifierInfo &II = AST.Idents.get("Load", tok::TokenKind::identifier);
    DeclarationName Load(&II);
    // TODO: We also need versions with status for CheckAccessFullyMapped.
    addHandleAccessFunction(Load, /*IsConst=*/false, /*IsRef=*/false);

    return *this;
  }

  FieldDecl *getResourceHandleField() {
    auto I = Fields.find("__handle");
    assert(I != Fields.end() &&
           I->second->getType()->isHLSLAttributedResourceType() &&
           "record does not have resource handle field");
    return I->second;
  }

  QualType getFirstTemplateTypeParam() {
    assert(Template && "record it not a template");
    if (const auto *TTD = dyn_cast<TemplateTypeParmDecl>(
            Template->getTemplateParameters()->getParam(0))) {
      return QualType(TTD->getTypeForDecl(), 0);
    }
    return QualType();
  }

  QualType getHandleElementType() {
    if (Template)
      return getFirstTemplateTypeParam();
    // TODO: Should we default to VoidTy? Using `i8` is arguably ambiguous.
    return SemaRef.getASTContext().Char8Ty;
  }

  BuiltinTypeDeclBuilder &startDefinition() {
    assert(!Record->isCompleteDefinition() && "record is already complete");
    Record->startDefinition();
    return *this;
  }

  BuiltinTypeDeclBuilder &completeDefinition() {
    assert(!Record->isCompleteDefinition() && "record is already complete");
    assert(Record->isBeingDefined() &&
           "Definition must be started before completing it.");

    Record->completeDefinition();
    return *this;
  }

  Expr *getConstantIntExpr(int value) {
    ASTContext &AST = SemaRef.getASTContext();
    return IntegerLiteral::Create(
        AST, llvm::APInt(AST.getTypeSize(AST.IntTy), value, true), AST.IntTy,
        SourceLocation());
  }

  TemplateParameterListBuilder addTemplateArgumentList();
  BuiltinTypeDeclBuilder &addSimpleTemplateParams(ArrayRef<StringRef> Names,
                                                  ConceptDecl *CD);

  // Builtin types methods
  BuiltinTypeDeclBuilder &addIncrementCounterMethod();
  BuiltinTypeDeclBuilder &addDecrementCounterMethod();
  BuiltinTypeDeclBuilder &addHandleAccessFunction(DeclarationName &Name,
                                                  bool IsConst, bool IsRef);
  BuiltinTypeDeclBuilder &addAppendMethod();
  BuiltinTypeDeclBuilder &addConsumeMethod();
};

struct TemplateParameterListBuilder {
  BuiltinTypeDeclBuilder &Builder;
  llvm::SmallVector<NamedDecl *> Params;

  TemplateParameterListBuilder(BuiltinTypeDeclBuilder &RB) : Builder(RB) {}

  ~TemplateParameterListBuilder() { finalizeTemplateArgs(); }

  TemplateParameterListBuilder &
  addTypeParameter(StringRef Name, QualType DefaultValue = QualType()) {
    assert(!Builder.Record->isCompleteDefinition() &&
           "record is already complete");
    ASTContext &AST = Builder.SemaRef.getASTContext();
    unsigned Position = static_cast<unsigned>(Params.size());
    auto *Decl = TemplateTypeParmDecl::Create(
        AST, Builder.Record->getDeclContext(), SourceLocation(),
        SourceLocation(), /* TemplateDepth */ 0, Position,
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
  constructConceptSpecializationExpr(Sema &S, ConceptDecl *CD) {
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
    TemplateTypeParmDecl *ConceptTTPD = dyn_cast<TemplateTypeParmDecl>(
        CD->getTemplateParameters()->getParam(0));

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
    TemplateArgumentLoc TAL = S.getTrivialTemplateArgumentLoc(
        ConceptTA, QualType(), SourceLocation());

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

  BuiltinTypeDeclBuilder &finalizeTemplateArgs(ConceptDecl *CD = nullptr) {
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

    QualType T = Builder.Template->getInjectedClassNameSpecialization();
    T = AST.getInjectedClassNameType(Builder.Record, T);

    return Builder;
  }
};

// Builder for methods of builtin types. Allows adding methods to builtin types
// using the builder pattern like this:
//
//   BuiltinTypeMethodBuilder(RecordBuilder, "MethodName", ReturnType)
//       .addParam("param_name", Type, InOutModifier)
//       .callBuiltin("builtin_name", BuiltinParams...)
//       .finalizeMethod();
//
// The builder needs to have all of the method parameters before it can create
// a CXXMethodDecl. It collects them in addParam calls and when a first
// method that builds the body is called or when access to 'this` is needed it
// creates the CXXMethodDecl and ParmVarDecls instances. These can then be
// referenced from the body building methods. Destructor or an explicit call to
// finalizeMethod() will complete the method definition.
//
// The callBuiltin helper method accepts constants via `Expr *` or placeholder
// value arguments to indicate which function arguments to forward to the
// builtin.
//
// If the method that is being built has a non-void return type the
// finalizeMethod will create a return statent with the value of the last
// statement (unless the last statement is already a ReturnStmt).
struct BuiltinTypeMethodBuilder {
  struct MethodParam {
    const IdentifierInfo &NameII;
    QualType Ty;
    HLSLParamModifierAttr::Spelling Modifier;
    MethodParam(const IdentifierInfo &NameII, QualType Ty,
                HLSLParamModifierAttr::Spelling Modifier)
        : NameII(NameII), Ty(Ty), Modifier(Modifier) {}
  };

  BuiltinTypeDeclBuilder &DeclBuilder;
  DeclarationNameInfo NameInfo;
  QualType ReturnTy;
  CXXMethodDecl *Method;
  bool IsConst;
  llvm::SmallVector<MethodParam> Params;
  llvm::SmallVector<Stmt *> StmtsList;

  // Argument placeholders, inspired by std::placeholder. These are the indices
  // of arguments to forward to `callBuiltin` and other method builder methods.
  // Additional special values are:
  //   Handle   - refers to the resource handle.
  //   LastStmt - refers to the last statement in the method body; referencing
  //              LastStmt will remove the statement from the method body since
  //              it will be linked from the new expression being constructed.
  enum class PlaceHolder { _0, _1, _2, _3, Handle = 128, LastStmt };

  Expr *convertPlaceholder(PlaceHolder PH) {
    if (PH == PlaceHolder::Handle)
      return getResourceHandleExpr();

    if (PH == PlaceHolder::LastStmt) {
      assert(!StmtsList.empty() && "no statements in the list");
      Stmt *LastStmt = StmtsList.pop_back_val();
      assert(isa<ValueStmt>(LastStmt) &&
             "last statement does not have a value");
      return cast<ValueStmt>(LastStmt)->getExprStmt();
    }

    ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
    ParmVarDecl *ParamDecl = Method->getParamDecl(static_cast<unsigned>(PH));
    return DeclRefExpr::Create(
        AST, NestedNameSpecifierLoc(), SourceLocation(), ParamDecl, false,
        DeclarationNameInfo(ParamDecl->getDeclName(), SourceLocation()),
        ParamDecl->getType(), VK_PRValue);
  }
  Expr *convertPlaceholder(Expr *E) { return E; }

public:
  BuiltinTypeMethodBuilder(BuiltinTypeDeclBuilder &DB, DeclarationName &Name,
                           QualType ReturnTy, bool IsConst = false)
      : DeclBuilder(DB), NameInfo(DeclarationNameInfo(Name, SourceLocation())),
        ReturnTy(ReturnTy), Method(nullptr), IsConst(IsConst) {}

  BuiltinTypeMethodBuilder(BuiltinTypeDeclBuilder &DB, StringRef Name,
                           QualType ReturnTy, bool IsConst = false)
      : DeclBuilder(DB), ReturnTy(ReturnTy), Method(nullptr), IsConst(IsConst) {
    const IdentifierInfo &II =
        DB.SemaRef.getASTContext().Idents.get(Name, tok::TokenKind::identifier);
    NameInfo = DeclarationNameInfo(DeclarationName(&II), SourceLocation());
  }

  BuiltinTypeMethodBuilder &addParam(StringRef Name, QualType Ty,
                                     HLSLParamModifierAttr::Spelling Modifier =
                                         HLSLParamModifierAttr::Keyword_in) {
    assert(Method == nullptr && "Cannot add param, method already created");
    const IdentifierInfo &II = DeclBuilder.SemaRef.getASTContext().Idents.get(
        Name, tok::TokenKind::identifier);
    Params.emplace_back(II, Ty, Modifier);
    return *this;
  }

private:
  void createMethodDecl() {
    assert(Method == nullptr && "Method already created");

    // create method type
    ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
    SmallVector<QualType> ParamTypes;
    for (MethodParam &MP : Params)
      ParamTypes.emplace_back(MP.Ty);

    FunctionProtoType::ExtProtoInfo ExtInfo;
    if (IsConst)
      ExtInfo.TypeQuals.addConst();

    QualType MethodTy = AST.getFunctionType(ReturnTy, ParamTypes, ExtInfo);

    // create method decl
    auto *TSInfo = AST.getTrivialTypeSourceInfo(MethodTy, SourceLocation());
    Method =
        CXXMethodDecl::Create(AST, DeclBuilder.Record, SourceLocation(),
                              NameInfo, MethodTy, TSInfo, SC_None, false, false,
                              ConstexprSpecKind::Unspecified, SourceLocation());

    // create params & set them to the function prototype
    SmallVector<ParmVarDecl *> ParmDecls;
    auto FnProtoLoc =
        Method->getTypeSourceInfo()->getTypeLoc().getAs<FunctionProtoTypeLoc>();
    for (int I = 0, E = Params.size(); I != E; I++) {
      MethodParam &MP = Params[I];
      ParmVarDecl *Parm = ParmVarDecl::Create(
          AST, Method->getDeclContext(), SourceLocation(), SourceLocation(),
          &MP.NameII, MP.Ty,
          AST.getTrivialTypeSourceInfo(MP.Ty, SourceLocation()), SC_None,
          nullptr);
      if (MP.Modifier != HLSLParamModifierAttr::Keyword_in) {
        auto *Mod =
            HLSLParamModifierAttr::Create(AST, SourceRange(), MP.Modifier);
        Parm->addAttr(Mod);
      }
      ParmDecls.push_back(Parm);
      FnProtoLoc.setParam(I, Parm);
    }
    Method->setParams({ParmDecls});
  }

public:
  ~BuiltinTypeMethodBuilder() { finalizeMethod(); }

  BuiltinTypeMethodBuilder(const BuiltinTypeMethodBuilder &Other) = delete;
  BuiltinTypeMethodBuilder &
  operator=(const BuiltinTypeMethodBuilder &Other) = delete;

  Expr *getResourceHandleExpr() {
    // The first statement added to a method or access to 'this' creates the
    // declaration.
    if (!Method)
      createMethodDecl();

    ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
    CXXThisExpr *This = CXXThisExpr::Create(
        AST, SourceLocation(), Method->getFunctionObjectParameterType(), true);
    FieldDecl *HandleField = DeclBuilder.getResourceHandleField();
    return MemberExpr::CreateImplicit(AST, This, false, HandleField,
                                      HandleField->getType(), VK_LValue,
                                      OK_Ordinary);
  }

  template <typename... Ts>
  BuiltinTypeMethodBuilder &callBuiltin(StringRef BuiltinName,
                                        QualType ReturnType, Ts... ArgSpecs) {
    std::array<Expr *, sizeof...(ArgSpecs)> Args{
        convertPlaceholder(std::forward<Ts>(ArgSpecs))...};

    // The first statement added to a method or access to 'this` creates the
    // declaration.
    if (!Method)
      createMethodDecl();

    ASTContext &AST = DeclBuilder.SemaRef.getASTContext();
    FunctionDecl *FD = lookupBuiltinFunction(DeclBuilder.SemaRef, BuiltinName);
    DeclRefExpr *DRE = DeclRefExpr::Create(
        AST, NestedNameSpecifierLoc(), SourceLocation(), FD, false,
        FD->getNameInfo(), AST.BuiltinFnTy, VK_PRValue);

    if (ReturnType.isNull())
      ReturnType = FD->getReturnType();

    Expr *Call = CallExpr::Create(AST, DRE, Args, ReturnType, VK_PRValue,
                                  SourceLocation(), FPOptionsOverride());
    StmtsList.push_back(Call);
    return *this;
  }

  template <typename TLHS, typename TRHS>
  BuiltinTypeMethodBuilder &assign(TLHS LHS, TRHS RHS) {
    Expr *LHSExpr = convertPlaceholder(LHS);
    Expr *RHSExpr = convertPlaceholder(RHS);
    Stmt *AssignStmt = BinaryOperator::Create(
        DeclBuilder.SemaRef.getASTContext(), LHSExpr, RHSExpr, BO_Assign,
        LHSExpr->getType(), ExprValueKind::VK_PRValue,
        ExprObjectKind::OK_Ordinary, SourceLocation(), FPOptionsOverride());
    StmtsList.push_back(AssignStmt);
    return *this;
  }

  template <typename T> BuiltinTypeMethodBuilder &dereference(T Ptr) {
    Expr *PtrExpr = convertPlaceholder(Ptr);
    Expr *Deref =
        UnaryOperator::Create(DeclBuilder.SemaRef.getASTContext(), PtrExpr,
                              UO_Deref, PtrExpr->getType()->getPointeeType(),
                              VK_PRValue, OK_Ordinary, SourceLocation(),
                              /*CanOverflow=*/false, FPOptionsOverride());
    StmtsList.push_back(Deref);
    return *this;
  }

  BuiltinTypeDeclBuilder &finalizeMethod() {
    assert(!DeclBuilder.Record->isCompleteDefinition() &&
           "record is already complete");
    assert(
        Method != nullptr &&
        "method decl not created; are you missing a call to build the body?");

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
      Method->setAccess(AccessSpecifier::AS_public);
      Method->addAttr(AlwaysInlineAttr::CreateImplicit(
          AST, SourceRange(), AlwaysInlineAttr::CXX11_clang_always_inline));
      DeclBuilder.Record->addDecl(Method);
    }
    return DeclBuilder;
  }
};

} // namespace

TemplateParameterListBuilder BuiltinTypeDeclBuilder::addTemplateArgumentList() {
  return TemplateParameterListBuilder(*this);
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

  TemplateParameterListBuilder Builder = this->addTemplateArgumentList();
  for (StringRef Name : Names)
    Builder.addTypeParameter(Name);
  return Builder.finalizeTemplateArgs(CD);
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addIncrementCounterMethod() {
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  return BuiltinTypeMethodBuilder(*this, "IncrementCounter",
                                  SemaRef.getASTContext().UnsignedIntTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", QualType(),
                   PH::Handle, getConstantIntExpr(1))
      .finalizeMethod();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addDecrementCounterMethod() {
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  return BuiltinTypeMethodBuilder(*this, "DecrementCounter",
                                  SemaRef.getASTContext().UnsignedIntTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", QualType(),
                   PH::Handle, getConstantIntExpr(-1))
      .finalizeMethod();
}

BuiltinTypeDeclBuilder &
BuiltinTypeDeclBuilder::addHandleAccessFunction(DeclarationName &Name,
                                                bool IsConst, bool IsRef) {
  assert(!Record->isCompleteDefinition() && "record is already complete");
  ASTContext &AST = SemaRef.getASTContext();
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;

  QualType ElemTy = getHandleElementType();
  // TODO: Map to an hlsl_device address space.
  QualType ElemPtrTy = AST.getPointerType(ElemTy);
  QualType ReturnTy = ElemTy;
  if (IsConst)
    ReturnTy.addConst();
  if (IsRef)
    ReturnTy = AST.getLValueReferenceType(ReturnTy);

  return BuiltinTypeMethodBuilder(*this, Name, ReturnTy, IsConst)
      .addParam("Index", AST.UnsignedIntTy)
      .callBuiltin("__builtin_hlsl_resource_getpointer", ElemPtrTy, PH::Handle,
                   PH::_0)
      .dereference(PH::LastStmt)
      .finalizeMethod();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addAppendMethod() {
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  ASTContext &AST = SemaRef.getASTContext();
  QualType ElemTy = getHandleElementType();
  return BuiltinTypeMethodBuilder(*this, "Append", AST.VoidTy)
      .addParam("value", ElemTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", AST.UnsignedIntTy,
                   PH::Handle, getConstantIntExpr(1))
      .callBuiltin("__builtin_hlsl_resource_getpointer",
                   AST.getPointerType(ElemTy), PH::Handle, PH::LastStmt)
      .dereference(PH::LastStmt)
      .assign(PH::LastStmt, PH::_0)
      .finalizeMethod();
}

BuiltinTypeDeclBuilder &BuiltinTypeDeclBuilder::addConsumeMethod() {
  using PH = BuiltinTypeMethodBuilder::PlaceHolder;
  ASTContext &AST = SemaRef.getASTContext();
  QualType ElemTy = getHandleElementType();
  return BuiltinTypeMethodBuilder(*this, "Consume", ElemTy)
      .callBuiltin("__builtin_hlsl_buffer_update_counter", AST.UnsignedIntTy,
                   PH::Handle, getConstantIntExpr(-1))
      .callBuiltin("__builtin_hlsl_resource_getpointer",
                   AST.getPointerType(ElemTy), PH::Handle, PH::LastStmt)
      .dereference(PH::LastStmt)
      .finalizeMethod();
}

HLSLExternalSemaSource::~HLSLExternalSemaSource() {}

void HLSLExternalSemaSource::InitializeSema(Sema &S) {
  SemaPtr = &S;
  ASTContext &AST = SemaPtr->getASTContext();
  // If the translation unit has external storage force external decls to load.
  if (AST.getTranslationUnitDecl()->hasExternalLexicalStorage())
    (void)AST.getTranslationUnitDecl()->decls_begin();

  IdentifierInfo &HLSL = AST.Idents.get("hlsl", tok::TokenKind::identifier);
  LookupResult Result(S, &HLSL, SourceLocation(), Sema::LookupNamespaceName);
  NamespaceDecl *PrevDecl = nullptr;
  if (S.LookupQualifiedName(Result, AST.getTranslationUnitDecl()))
    PrevDecl = Result.getAsSingle<NamespaceDecl>();
  HLSLNamespace = NamespaceDecl::Create(
      AST, AST.getTranslationUnitDecl(), /*Inline=*/false, SourceLocation(),
      SourceLocation(), &HLSL, PrevDecl, /*Nested=*/false);
  HLSLNamespace->setImplicit(true);
  HLSLNamespace->setHasExternalLexicalStorage();
  AST.getTranslationUnitDecl()->addDecl(HLSLNamespace);

  // Force external decls in the HLSL namespace to load from the PCH.
  (void)HLSLNamespace->getCanonicalDecl()->decls_begin();
  defineTrivialHLSLTypes();
  defineHLSLTypesWithForwardDeclarations();

  // This adds a `using namespace hlsl` directive. In DXC, we don't put HLSL's
  // built in types inside a namespace, but we are planning to change that in
  // the near future. In order to be source compatible older versions of HLSL
  // will need to implicitly use the hlsl namespace. For now in clang everything
  // will get added to the namespace, and we can remove the using directive for
  // future language versions to match HLSL's evolution.
  auto *UsingDecl = UsingDirectiveDecl::Create(
      AST, AST.getTranslationUnitDecl(), SourceLocation(), SourceLocation(),
      NestedNameSpecifierLoc(), SourceLocation(), HLSLNamespace,
      AST.getTranslationUnitDecl());

  AST.getTranslationUnitDecl()->addDecl(UsingDecl);
}

void HLSLExternalSemaSource::defineHLSLVectorAlias() {
  ASTContext &AST = SemaPtr->getASTContext();

  llvm::SmallVector<NamedDecl *> TemplateParams;

  auto *TypeParam = TemplateTypeParmDecl::Create(
      AST, HLSLNamespace, SourceLocation(), SourceLocation(), 0, 0,
      &AST.Idents.get("element", tok::TokenKind::identifier), false, false);
  TypeParam->setDefaultArgument(
      AST, SemaPtr->getTrivialTemplateArgumentLoc(
               TemplateArgument(AST.FloatTy), QualType(), SourceLocation()));

  TemplateParams.emplace_back(TypeParam);

  auto *SizeParam = NonTypeTemplateParmDecl::Create(
      AST, HLSLNamespace, SourceLocation(), SourceLocation(), 0, 1,
      &AST.Idents.get("element_count", tok::TokenKind::identifier), AST.IntTy,
      false, AST.getTrivialTypeSourceInfo(AST.IntTy));
  llvm::APInt Val(AST.getIntWidth(AST.IntTy), 4);
  TemplateArgument Default(AST, llvm::APSInt(std::move(Val)), AST.IntTy,
                           /*IsDefaulted=*/true);
  SizeParam->setDefaultArgument(
      AST, SemaPtr->getTrivialTemplateArgumentLoc(Default, AST.IntTy,
                                                  SourceLocation(), SizeParam));
  TemplateParams.emplace_back(SizeParam);

  auto *ParamList =
      TemplateParameterList::Create(AST, SourceLocation(), SourceLocation(),
                                    TemplateParams, SourceLocation(), nullptr);

  IdentifierInfo &II = AST.Idents.get("vector", tok::TokenKind::identifier);

  QualType AliasType = AST.getDependentSizedExtVectorType(
      AST.getTemplateTypeParmType(0, 0, false, TypeParam),
      DeclRefExpr::Create(
          AST, NestedNameSpecifierLoc(), SourceLocation(), SizeParam, false,
          DeclarationNameInfo(SizeParam->getDeclName(), SourceLocation()),
          AST.IntTy, VK_LValue),
      SourceLocation());

  auto *Record = TypeAliasDecl::Create(AST, HLSLNamespace, SourceLocation(),
                                       SourceLocation(), &II,
                                       AST.getTrivialTypeSourceInfo(AliasType));
  Record->setImplicit(true);

  auto *Template =
      TypeAliasTemplateDecl::Create(AST, HLSLNamespace, SourceLocation(),
                                    Record->getIdentifier(), ParamList, Record);

  Record->setDescribedAliasTemplate(Template);
  Template->setImplicit(true);
  Template->setLexicalDeclContext(Record->getDeclContext());
  HLSLNamespace->addDecl(Template);
}

void HLSLExternalSemaSource::defineTrivialHLSLTypes() {
  defineHLSLVectorAlias();
}

/// Set up common members and attributes for buffer types
static BuiltinTypeDeclBuilder setupBufferType(CXXRecordDecl *Decl, Sema &S,
                                              ResourceClass RC, ResourceKind RK,
                                              bool IsROV, bool RawBuffer) {
  return BuiltinTypeDeclBuilder(S, Decl)
      .addHandleMember(RC, RK, IsROV, RawBuffer)
      .addDefaultHandleConstructor();
}

// This function is responsible for constructing the constraint expression for
// this concept:
// template<typename T> concept is_typed_resource_element_compatible =
// __is_typed_resource_element_compatible<T>;
static Expr *constructTypedBufferConstraintExpr(Sema &S, SourceLocation NameLoc,
                                                TemplateTypeParmDecl *T) {
  ASTContext &Context = S.getASTContext();

  // Obtain the QualType for 'bool'
  QualType BoolTy = Context.BoolTy;

  // Create a QualType that points to this TemplateTypeParmDecl
  QualType TType = Context.getTypeDeclType(T);

  // Create a TypeSourceInfo for the template type parameter 'T'
  TypeSourceInfo *TTypeSourceInfo =
      Context.getTrivialTypeSourceInfo(TType, NameLoc);

  TypeTraitExpr *TypedResExpr = TypeTraitExpr::Create(
      Context, BoolTy, NameLoc, UTT_IsTypedResourceElementCompatible,
      {TTypeSourceInfo}, NameLoc, true);

  return TypedResExpr;
}

// This function is responsible for constructing the constraint expression for
// this concept:
// template<typename T> concept is_structured_resource_element_compatible =
// !__is_intangible<T> && sizeof(T) >= 1;
static Expr *constructStructuredBufferConstraintExpr(Sema &S,
                                                     SourceLocation NameLoc,
                                                     TemplateTypeParmDecl *T) {
  ASTContext &Context = S.getASTContext();

  // Obtain the QualType for 'bool'
  QualType BoolTy = Context.BoolTy;

  // Create a QualType that points to this TemplateTypeParmDecl
  QualType TType = Context.getTypeDeclType(T);

  // Create a TypeSourceInfo for the template type parameter 'T'
  TypeSourceInfo *TTypeSourceInfo =
      Context.getTrivialTypeSourceInfo(TType, NameLoc);

  TypeTraitExpr *IsIntangibleExpr =
      TypeTraitExpr::Create(Context, BoolTy, NameLoc, UTT_IsIntangibleType,
                            {TTypeSourceInfo}, NameLoc, true);

  // negate IsIntangibleExpr
  UnaryOperator *NotIntangibleExpr = UnaryOperator::Create(
      Context, IsIntangibleExpr, UO_LNot, BoolTy, VK_LValue, OK_Ordinary,
      NameLoc, false, FPOptionsOverride());

  // element types also may not be of 0 size
  UnaryExprOrTypeTraitExpr *SizeOfExpr = new (Context) UnaryExprOrTypeTraitExpr(
      UETT_SizeOf, TTypeSourceInfo, BoolTy, NameLoc, NameLoc);

  // Create a BinaryOperator that checks if the size of the type is not equal to
  // 1 Empty structs have a size of 1 in HLSL, so we need to check for that
  IntegerLiteral *rhs = IntegerLiteral::Create(
      Context, llvm::APInt(Context.getTypeSize(Context.getSizeType()), 1, true),
      Context.getSizeType(), NameLoc);

  BinaryOperator *SizeGEQOneExpr =
      BinaryOperator::Create(Context, SizeOfExpr, rhs, BO_GE, BoolTy, VK_LValue,
                             OK_Ordinary, NameLoc, FPOptionsOverride());

  // Combine the two constraints
  BinaryOperator *CombinedExpr = BinaryOperator::Create(
      Context, NotIntangibleExpr, SizeGEQOneExpr, BO_LAnd, BoolTy, VK_LValue,
      OK_Ordinary, NameLoc, FPOptionsOverride());

  return CombinedExpr;
}

static ConceptDecl *constructBufferConceptDecl(Sema &S, NamespaceDecl *NSD,
                                               bool isTypedBuffer) {
  ASTContext &Context = S.getASTContext();
  DeclContext *DC = NSD->getDeclContext();
  SourceLocation DeclLoc = SourceLocation();

  IdentifierInfo &ElementTypeII = Context.Idents.get("element_type");
  TemplateTypeParmDecl *T = TemplateTypeParmDecl::Create(
      Context, NSD->getDeclContext(), DeclLoc, DeclLoc,
      /*D=*/0,
      /*P=*/0,
      /*Id=*/&ElementTypeII,
      /*Typename=*/true,
      /*ParameterPack=*/false);

  T->setDeclContext(DC);
  T->setReferenced();

  // Create and Attach Template Parameter List to ConceptDecl
  TemplateParameterList *ConceptParams = TemplateParameterList::Create(
      Context, DeclLoc, DeclLoc, {T}, DeclLoc, nullptr);

  DeclarationName DeclName;
  Expr *ConstraintExpr = nullptr;

  if (isTypedBuffer) {
    DeclName = DeclarationName(
        &Context.Idents.get("__is_typed_resource_element_compatible"));
    ConstraintExpr = constructTypedBufferConstraintExpr(S, DeclLoc, T);
  } else {
    DeclName = DeclarationName(
        &Context.Idents.get("__is_structured_resource_element_compatible"));
    ConstraintExpr = constructStructuredBufferConstraintExpr(S, DeclLoc, T);
  }

  // Create a ConceptDecl
  ConceptDecl *CD =
      ConceptDecl::Create(Context, NSD->getDeclContext(), DeclLoc, DeclName,
                          ConceptParams, ConstraintExpr);

  // Attach the template parameter list to the ConceptDecl
  CD->setTemplateParameters(ConceptParams);

  // Add the concept declaration to the Translation Unit Decl
  NSD->getDeclContext()->addDecl(CD);

  return CD;
}

void HLSLExternalSemaSource::defineHLSLTypesWithForwardDeclarations() {
  CXXRecordDecl *Decl;
  ConceptDecl *TypedBufferConcept = constructBufferConceptDecl(
      *SemaPtr, HLSLNamespace, /*isTypedBuffer*/ true);
  ConceptDecl *StructuredBufferConcept = constructBufferConceptDecl(
      *SemaPtr, HLSLNamespace, /*isTypedBuffer*/ false);
  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RWBuffer")
             .addSimpleTemplateParams({"element_type"}, TypedBufferConcept)
             .finalizeForwardDeclaration();

  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV,
                    ResourceKind::TypedBuffer, /*IsROV=*/false,
                    /*RawBuffer=*/false)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .completeDefinition();
  });

  Decl =
      BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RasterizerOrderedBuffer")
          .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
          .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV,
                    ResourceKind::TypedBuffer, /*IsROV=*/true,
                    /*RawBuffer=*/false)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "StructuredBuffer")
             .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::SRV, ResourceKind::RawBuffer,
                    /*IsROV=*/false, /*RawBuffer=*/true)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RWStructuredBuffer")
             .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, ResourceKind::RawBuffer,
                    /*IsROV=*/false, /*RawBuffer=*/true)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .addIncrementCounterMethod()
        .addDecrementCounterMethod()
        .completeDefinition();
  });

  Decl =
      BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "AppendStructuredBuffer")
          .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
          .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, ResourceKind::RawBuffer,
                    /*IsROV=*/false, /*RawBuffer=*/true)
        .addAppendMethod()
        .completeDefinition();
  });

  Decl =
      BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "ConsumeStructuredBuffer")
          .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
          .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, ResourceKind::RawBuffer,
                    /*IsROV=*/false, /*RawBuffer=*/true)
        .addConsumeMethod()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace,
                                "RasterizerOrderedStructuredBuffer")
             .addSimpleTemplateParams({"element_type"}, StructuredBufferConcept)
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, ResourceKind::RawBuffer,
                    /*IsROV=*/true, /*RawBuffer=*/true)
        .addArraySubscriptOperators()
        .addLoadMethods()
        .addIncrementCounterMethod()
        .addDecrementCounterMethod()
        .completeDefinition();
  });

  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "ByteAddressBuffer")
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::SRV, ResourceKind::RawBuffer,
                    /*IsROV=*/false,
                    /*RawBuffer=*/true)
        .completeDefinition();
  });
  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace, "RWByteAddressBuffer")
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, ResourceKind::RawBuffer,
                    /*IsROV=*/false,
                    /*RawBuffer=*/true)
        .completeDefinition();
  });
  Decl = BuiltinTypeDeclBuilder(*SemaPtr, HLSLNamespace,
                                "RasterizerOrderedByteAddressBuffer")
             .finalizeForwardDeclaration();
  onCompletion(Decl, [this](CXXRecordDecl *Decl) {
    setupBufferType(Decl, *SemaPtr, ResourceClass::UAV, ResourceKind::RawBuffer,
                    /*IsROV=*/true,
                    /*RawBuffer=*/true)
        .completeDefinition();
  });
}

void HLSLExternalSemaSource::onCompletion(CXXRecordDecl *Record,
                                          CompletionFunction Fn) {
  if (!Record->isCompleteDefinition())
    Completions.insert(std::make_pair(Record->getCanonicalDecl(), Fn));
}

void HLSLExternalSemaSource::CompleteType(TagDecl *Tag) {
  if (!isa<CXXRecordDecl>(Tag))
    return;
  auto Record = cast<CXXRecordDecl>(Tag);

  // If this is a specialization, we need to get the underlying templated
  // declaration and complete that.
  if (auto TDecl = dyn_cast<ClassTemplateSpecializationDecl>(Record))
    Record = TDecl->getSpecializedTemplate()->getTemplatedDecl();
  Record = Record->getCanonicalDecl();
  auto It = Completions.find(Record);
  if (It == Completions.end())
    return;
  It->second(Record);
}

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
