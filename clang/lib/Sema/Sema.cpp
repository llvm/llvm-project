//===--- Sema.cpp - AST Builder and Semantic Analysis Implementation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the actions class which performs semantic analysis and
// builds an AST out of a parse stream.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "llvm/ADT/DenseMap.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/TargetInfo.h"
using namespace clang;

/// ConvertQualTypeToStringFn - This function is used to pretty print the 
/// specified QualType as a string in diagnostics.
static void ConvertArgToStringFn(Diagnostic::ArgumentKind Kind, intptr_t Val,
                                 const char *Modifier, unsigned ModLen,
                                 const char *Argument, unsigned ArgLen,
                                 llvm::SmallVectorImpl<char> &Output,
                                 void *Cookie) {
  ASTContext &Context = *static_cast<ASTContext*>(Cookie);
  
  std::string S;
  if (Kind == Diagnostic::ak_qualtype) {
    assert(ModLen == 0 && ArgLen == 0 &&
           "Invalid modifier for QualType argument");

    QualType Ty(QualType::getFromOpaquePtr(reinterpret_cast<void*>(Val)));

    // FIXME: Playing with std::string is really slow.
    S = Ty.getAsString(Context.PrintingPolicy);
    
    // If this is a sugared type (like a typedef, typeof, etc), then unwrap one
    // level of the sugar so that the type is more obvious to the user.
    QualType DesugaredTy = Ty->getDesugaredType(true);
    DesugaredTy.setCVRQualifiers(DesugaredTy.getCVRQualifiers() |
                                 Ty.getCVRQualifiers());

    if (Ty != DesugaredTy &&
        // If the desugared type is a vector type, we don't want to expand it,
        // it will turn into an attribute mess. People want their "vec4".
        !isa<VectorType>(DesugaredTy) &&
      
        // Don't desugar magic Objective-C types.
        Ty.getUnqualifiedType() != Context.getObjCIdType() &&
        Ty.getUnqualifiedType() != Context.getObjCClassType() &&
        Ty.getUnqualifiedType() != Context.getObjCSelType() &&
        Ty.getUnqualifiedType() != Context.getObjCProtoType() &&
        
        // Not va_list.
        Ty.getUnqualifiedType() != Context.getBuiltinVaListType()) {
      S = "'"+S+"' (aka '";
      S += DesugaredTy.getAsString(Context.PrintingPolicy);
      S += "')";
      Output.append(S.begin(), S.end());
      return;
    }
    
  } else if (Kind == Diagnostic::ak_declarationname) {
   
    DeclarationName N = DeclarationName::getFromOpaqueInteger(Val);
    S = N.getAsString();
    
    if (ModLen == 9 && !memcmp(Modifier, "objcclass", 9) && ArgLen == 0)
      S = '+' + S;
    else if (ModLen == 12 && !memcmp(Modifier, "objcinstance", 12) && ArgLen==0)
      S = '-' + S;
    else
      assert(ModLen == 0 && ArgLen == 0 &&
             "Invalid modifier for DeclarationName argument");
  } else {
    assert(Kind == Diagnostic::ak_nameddecl);
    if (ModLen == 1 && Modifier[0] == 'q' && ArgLen == 0)
      S = reinterpret_cast<NamedDecl*>(Val)->getQualifiedNameAsString();
    else { 
      assert(ModLen == 0 && ArgLen == 0 &&
           "Invalid modifier for NamedDecl* argument");
      S = reinterpret_cast<NamedDecl*>(Val)->getNameAsString();
    }
  }
  
  Output.push_back('\'');
  Output.append(S.begin(), S.end());
  Output.push_back('\'');
}


static inline RecordDecl *CreateStructDecl(ASTContext &C, const char *Name) {
  if (C.getLangOptions().CPlusPlus)
    return CXXRecordDecl::Create(C, TagDecl::TK_struct, 
                                 C.getTranslationUnitDecl(),
                                 SourceLocation(), &C.Idents.get(Name));

  return RecordDecl::Create(C, TagDecl::TK_struct, 
                            C.getTranslationUnitDecl(),
                            SourceLocation(), &C.Idents.get(Name));
}

void Sema::ActOnTranslationUnitScope(SourceLocation Loc, Scope *S) {
  TUScope = S;
  PushDeclContext(S, Context.getTranslationUnitDecl());
  
  if (PP.getTargetInfo().getPointerWidth(0) >= 64) {
    // Install [u]int128_t for 64-bit targets.
    PushOnScopeChains(TypedefDecl::Create(Context, CurContext,
                                          SourceLocation(),
                                          &Context.Idents.get("__int128_t"),
                                          Context.Int128Ty), TUScope);
    PushOnScopeChains(TypedefDecl::Create(Context, CurContext,
                                          SourceLocation(),
                                          &Context.Idents.get("__uint128_t"),
                                          Context.UnsignedInt128Ty), TUScope);
  }
  
  
  if (!PP.getLangOptions().ObjC1) return;
  
  // Built-in ObjC types may already be set by PCHReader (hence isNull checks).
  if (Context.getObjCSelType().isNull()) {
    // Synthesize "typedef struct objc_selector *SEL;"
    RecordDecl *SelTag = CreateStructDecl(Context, "objc_selector");
    PushOnScopeChains(SelTag, TUScope);
  
    QualType SelT = Context.getPointerType(Context.getTagDeclType(SelTag));
    TypedefDecl *SelTypedef = TypedefDecl::Create(Context, CurContext,
                                                  SourceLocation(),
                                                  &Context.Idents.get("SEL"),
                                                  SelT);
    PushOnScopeChains(SelTypedef, TUScope);
    Context.setObjCSelType(Context.getTypeDeclType(SelTypedef));
  }

  // Synthesize "@class Protocol;
  if (Context.getObjCProtoType().isNull()) {
    ObjCInterfaceDecl *ProtocolDecl =
      ObjCInterfaceDecl::Create(Context, CurContext, SourceLocation(),
                                &Context.Idents.get("Protocol"), 
                                SourceLocation(), true);
    Context.setObjCProtoType(Context.getObjCInterfaceType(ProtocolDecl));
    PushOnScopeChains(ProtocolDecl, TUScope);
  }
  // Create the built-in typedef for 'id'.
  if (Context.getObjCIdType().isNull()) {
    TypedefDecl *IdTypedef = 
      TypedefDecl::Create( 
        Context, CurContext, SourceLocation(), &Context.Idents.get("id"),
        Context.getObjCObjectPointerType(Context.ObjCBuiltinIdTy)
      );
    PushOnScopeChains(IdTypedef, TUScope);
    Context.setObjCIdType(Context.getTypeDeclType(IdTypedef));
    Context.ObjCIdRedefinitionType = Context.getObjCIdType();
  }
  // Create the built-in typedef for 'Class'.
  if (Context.getObjCClassType().isNull()) {
    TypedefDecl *ClassTypedef = 
      TypedefDecl::Create( 
        Context, CurContext, SourceLocation(), &Context.Idents.get("Class"),
        Context.getObjCObjectPointerType(Context.ObjCBuiltinClassTy)
      );
    PushOnScopeChains(ClassTypedef, TUScope);
    Context.setObjCClassType(Context.getTypeDeclType(ClassTypedef));
    Context.ObjCClassRedefinitionType = Context.getObjCClassType();
  }
}

Sema::Sema(Preprocessor &pp, ASTContext &ctxt, ASTConsumer &consumer,
           bool CompleteTranslationUnit)
  : LangOpts(pp.getLangOptions()), PP(pp), Context(ctxt), Consumer(consumer),
    Diags(PP.getDiagnostics()), SourceMgr(PP.getSourceManager()), 
    ExternalSource(0), CurContext(0), PreDeclaratorDC(0),
    CurBlock(0), PackContext(0), IdResolver(pp.getLangOptions()),
    GlobalNewDeleteDeclared(false), ExprEvalContext(PotentiallyEvaluated),
    CompleteTranslationUnit(CompleteTranslationUnit),
    NumSFINAEErrors(0), CurrentInstantiationScope(0) {
  
  StdNamespace = 0;
  TUScope = 0;
  if (getLangOptions().CPlusPlus)
    FieldCollector.reset(new CXXFieldCollector());
      
  // Tell diagnostics how to render things from the AST library.
  PP.getDiagnostics().SetArgToStringFn(ConvertArgToStringFn, &Context);
}

/// ImpCastExprToType - If Expr is not of type 'Type', insert an implicit cast. 
/// If there is already an implicit cast, merge into the existing one.
/// If isLvalue, the result of the cast is an lvalue.
void Sema::ImpCastExprToType(Expr *&Expr, QualType Ty, 
                             const CastExpr::CastInfo &Info, bool isLvalue) {
  QualType ExprTy = Context.getCanonicalType(Expr->getType());
  QualType TypeTy = Context.getCanonicalType(Ty);
  
  if (ExprTy == TypeTy)
    return;
  
  if (Expr->getType().getTypePtr()->isPointerType() &&
      Ty.getTypePtr()->isPointerType()) {
    QualType ExprBaseType = 
      cast<PointerType>(ExprTy.getUnqualifiedType())->getPointeeType();
    QualType BaseType =
      cast<PointerType>(TypeTy.getUnqualifiedType())->getPointeeType();
    if (ExprBaseType.getAddressSpace() != BaseType.getAddressSpace()) {
      Diag(Expr->getExprLoc(), diag::err_implicit_pointer_address_space_cast)
        << Expr->getSourceRange();
    }
  }
  
  if (ImplicitCastExpr *ImpCast = dyn_cast<ImplicitCastExpr>(Expr)) {
    ImpCast->setType(Ty);
    ImpCast->setLvalueCast(isLvalue);
  } else 
    Expr = new (Context) ImplicitCastExpr(Ty, Info, Expr, 
                                          isLvalue);
}

void Sema::DeleteExpr(ExprTy *E) {
  if (E) static_cast<Expr*>(E)->Destroy(Context);
}
void Sema::DeleteStmt(StmtTy *S) {
  if (S) static_cast<Stmt*>(S)->Destroy(Context);
}

/// ActOnEndOfTranslationUnit - This is called at the very end of the
/// translation unit when EOF is reached and all but the top-level scope is
/// popped.
void Sema::ActOnEndOfTranslationUnit() {
  // C++: Perform implicit template instantiations.
  //
  // FIXME: When we perform these implicit instantiations, we do not carefully
  // keep track of the point of instantiation (C++ [temp.point]). This means
  // that name lookup that occurs within the template instantiation will
  // always happen at the end of the translation unit, so it will find
  // some names that should not be found. Although this is common behavior 
  // for C++ compilers, it is technically wrong. In the future, we either need
  // to be able to filter the results of name lookup or we need to perform
  // template instantiations earlier.
  PerformPendingImplicitInstantiations();
  
  // Check for #pragma weak identifiers that were never declared
  // FIXME: This will cause diagnostics to be emitted in a non-determinstic
  // order!  Iterating over a densemap like this is bad.
  for (llvm::DenseMap<IdentifierInfo*,WeakInfo>::iterator
       I = WeakUndeclaredIdentifiers.begin(),
       E = WeakUndeclaredIdentifiers.end(); I != E; ++I) {
    if (I->second.getUsed()) continue;
    
    Diag(I->second.getLocation(), diag::warn_weak_identifier_undeclared)
      << I->first;
  }

  if (!CompleteTranslationUnit)
    return;

  // C99 6.9.2p2:
  //   A declaration of an identifier for an object that has file
  //   scope without an initializer, and without a storage-class
  //   specifier or with the storage-class specifier static,
  //   constitutes a tentative definition. If a translation unit
  //   contains one or more tentative definitions for an identifier,
  //   and the translation unit contains no external definition for
  //   that identifier, then the behavior is exactly as if the
  //   translation unit contains a file scope declaration of that
  //   identifier, with the composite type as of the end of the
  //   translation unit, with an initializer equal to 0.
  for (unsigned i = 0, e = TentativeDefinitionList.size(); i != e; ++i) {
    VarDecl *VD = TentativeDefinitions.lookup(TentativeDefinitionList[i]);
    
    // If the tentative definition was completed, it will be in the list, but
    // not the map.
    if (VD == 0 || VD->isInvalidDecl() || !VD->isTentativeDefinition(Context))
      continue;

    if (const IncompleteArrayType *ArrayT 
        = Context.getAsIncompleteArrayType(VD->getType())) {
      if (RequireCompleteType(VD->getLocation(), 
                              ArrayT->getElementType(),
                              diag::err_tentative_def_incomplete_type_arr)) {
        VD->setInvalidDecl();
        continue;
      }
      
      // Set the length of the array to 1 (C99 6.9.2p5).
      Diag(VD->getLocation(), diag::warn_tentative_incomplete_array);
      llvm::APInt One(Context.getTypeSize(Context.getSizeType()), true);
      QualType T 
        = Context.getConstantArrayWithoutExprType(ArrayT->getElementType(),
                                                  One, ArrayType::Normal, 0);
      VD->setType(T);
    } else if (RequireCompleteType(VD->getLocation(), VD->getType(), 
                                   diag::err_tentative_def_incomplete_type))
      VD->setInvalidDecl();

    // Notify the consumer that we've completed a tentative definition.
    if (!VD->isInvalidDecl())
      Consumer.CompleteTentativeDefinition(VD);

  }
}


//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

DeclContext *Sema::getFunctionLevelDeclContext() {
  DeclContext *DC = PreDeclaratorDC ? PreDeclaratorDC : CurContext;
  
  while (isa<BlockDecl>(DC))
    DC = DC->getParent();
  
  return DC;
}

/// getCurFunctionDecl - If inside of a function body, this returns a pointer
/// to the function decl for the function being parsed.  If we're currently
/// in a 'block', this returns the containing context.
FunctionDecl *Sema::getCurFunctionDecl() {
  DeclContext *DC = getFunctionLevelDeclContext();
  return dyn_cast<FunctionDecl>(DC);
}

ObjCMethodDecl *Sema::getCurMethodDecl() {
  DeclContext *DC = getFunctionLevelDeclContext();
  return dyn_cast<ObjCMethodDecl>(DC);
}

NamedDecl *Sema::getCurFunctionOrMethodDecl() {
  DeclContext *DC = getFunctionLevelDeclContext();
  if (isa<ObjCMethodDecl>(DC) || isa<FunctionDecl>(DC))
    return cast<NamedDecl>(DC);
  return 0;
}

Sema::SemaDiagnosticBuilder::~SemaDiagnosticBuilder() {
  if (!this->Emit())
    return;
  
  // If this is not a note, and we're in a template instantiation
  // that is different from the last template instantiation where
  // we emitted an error, print a template instantiation
  // backtrace.
  if (!SemaRef.Diags.isBuiltinNote(DiagID) &&
      !SemaRef.ActiveTemplateInstantiations.empty() &&
      SemaRef.ActiveTemplateInstantiations.back() 
        != SemaRef.LastTemplateInstantiationErrorContext) {
    SemaRef.PrintInstantiationStack();
    SemaRef.LastTemplateInstantiationErrorContext 
      = SemaRef.ActiveTemplateInstantiations.back();
  }
}

void Sema::ActOnComment(SourceRange Comment) {
  Context.Comments.push_back(Comment);
}
