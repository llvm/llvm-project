//===--- SemaLambda.cpp - Semantic Analysis for C++11 Lambdas -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ lambda expressions.
//
//===----------------------------------------------------------------------===//
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/AST/ExprCXX.h"
using namespace clang;
using namespace sema;

CXXRecordDecl *Sema::createLambdaClosureType(SourceRange IntroducerRange,
                                             TypeSourceInfo *Info,
                                             bool KnownDependent) {
  DeclContext *DC = CurContext;
  while (!(DC->isFunctionOrMethod() || DC->isRecord() || DC->isFileContext()))
    DC = DC->getParent();
  
  // Start constructing the lambda class.
  CXXRecordDecl *Class = CXXRecordDecl::CreateLambda(Context, DC, Info,
                                                     IntroducerRange.getBegin(),
                                                     KnownDependent);
  DC->addDecl(Class);
  
  return Class;
}

/// \brief Determine whether the given context is or is enclosed in an inline
/// function.
static bool isInInlineFunction(const DeclContext *DC) {
  while (!DC->isFileContext()) {
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(DC))
      if (FD->isInlined())
        return true;
    
    DC = DC->getLexicalParent();
  }
  
  return false;
}

CXXMethodDecl *Sema::startLambdaDefinition(CXXRecordDecl *Class,
                 SourceRange IntroducerRange,
                 TypeSourceInfo *MethodType,
                 SourceLocation EndLoc,
                 llvm::ArrayRef<ParmVarDecl *> Params) {
  // C++11 [expr.prim.lambda]p5:
  //   The closure type for a lambda-expression has a public inline function 
  //   call operator (13.5.4) whose parameters and return type are described by
  //   the lambda-expression's parameter-declaration-clause and 
  //   trailing-return-type respectively.
  DeclarationName MethodName
    = Context.DeclarationNames.getCXXOperatorName(OO_Call);
  DeclarationNameLoc MethodNameLoc;
  MethodNameLoc.CXXOperatorName.BeginOpNameLoc
    = IntroducerRange.getBegin().getRawEncoding();
  MethodNameLoc.CXXOperatorName.EndOpNameLoc
    = IntroducerRange.getEnd().getRawEncoding();
  CXXMethodDecl *Method
    = CXXMethodDecl::Create(Context, Class, EndLoc,
                            DeclarationNameInfo(MethodName, 
                                                IntroducerRange.getBegin(),
                                                MethodNameLoc),
                            MethodType->getType(), MethodType,
                            /*isStatic=*/false,
                            SC_None,
                            /*isInline=*/true,
                            /*isConstExpr=*/false,
                            EndLoc);
  Method->setAccess(AS_public);
  
  // Temporarily set the lexical declaration context to the current
  // context, so that the Scope stack matches the lexical nesting.
  Method->setLexicalDeclContext(CurContext);  
  
  // Add parameters.
  if (!Params.empty()) {
    Method->setParams(Params);
    CheckParmsForFunctionDef(const_cast<ParmVarDecl **>(Params.begin()),
                             const_cast<ParmVarDecl **>(Params.end()),
                             /*CheckParameterNames=*/false);
    
    for (CXXMethodDecl::param_iterator P = Method->param_begin(), 
                                    PEnd = Method->param_end();
         P != PEnd; ++P)
      (*P)->setOwningFunction(Method);
  }

  // Allocate a mangling number for this lambda expression, if the ABI
  // requires one.
  Decl *ContextDecl = ExprEvalContexts.back().LambdaContextDecl;

  enum ContextKind {
    Normal,
    DefaultArgument,
    DataMember,
    StaticDataMember
  } Kind = Normal;

  // Default arguments of member function parameters that appear in a class
  // definition, as well as the initializers of data members, receive special
  // treatment. Identify them.
  if (ContextDecl) {
    if (ParmVarDecl *Param = dyn_cast<ParmVarDecl>(ContextDecl)) {
      if (const DeclContext *LexicalDC
          = Param->getDeclContext()->getLexicalParent())
        if (LexicalDC->isRecord())
          Kind = DefaultArgument;
    } else if (VarDecl *Var = dyn_cast<VarDecl>(ContextDecl)) {
      if (Var->getDeclContext()->isRecord())
        Kind = StaticDataMember;
    } else if (isa<FieldDecl>(ContextDecl)) {
      Kind = DataMember;
    }
  }

  // Itanium ABI [5.1.7]:
  //   In the following contexts [...] the one-definition rule requires closure
  //   types in different translation units to "correspond":
  bool IsInNonspecializedTemplate =
    !ActiveTemplateInstantiations.empty() || CurContext->isDependentContext();
  unsigned ManglingNumber;
  switch (Kind) {
  case Normal:
    //  -- the bodies of non-exported nonspecialized template functions
    //  -- the bodies of inline functions
    if ((IsInNonspecializedTemplate &&
         !(ContextDecl && isa<ParmVarDecl>(ContextDecl))) ||
        isInInlineFunction(CurContext))
      ManglingNumber = Context.getLambdaManglingNumber(Method);
    else
      ManglingNumber = 0;

    // There is no special context for this lambda.
    ContextDecl = 0;
    break;

  case StaticDataMember:
    //  -- the initializers of nonspecialized static members of template classes
    if (!IsInNonspecializedTemplate) {
      ManglingNumber = 0;
      ContextDecl = 0;
      break;
    }
    // Fall through to assign a mangling number.

  case DataMember:
    //  -- the in-class initializers of class members
  case DefaultArgument:
    //  -- default arguments appearing in class definitions
    ManglingNumber = ExprEvalContexts.back().getLambdaMangleContext()
                       .getManglingNumber(Method);
    break;
  }

  Class->setLambdaMangling(ManglingNumber, ContextDecl);

  return Method;
}

LambdaScopeInfo *Sema::enterLambdaScope(CXXMethodDecl *CallOperator,
                                        SourceRange IntroducerRange,
                                        LambdaCaptureDefault CaptureDefault,
                                        bool ExplicitParams,
                                        bool ExplicitResultType,
                                        bool Mutable) {
  PushLambdaScope(CallOperator->getParent(), CallOperator);
  LambdaScopeInfo *LSI = getCurLambda();
  if (CaptureDefault == LCD_ByCopy)
    LSI->ImpCaptureStyle = LambdaScopeInfo::ImpCap_LambdaByval;
  else if (CaptureDefault == LCD_ByRef)
    LSI->ImpCaptureStyle = LambdaScopeInfo::ImpCap_LambdaByref;
  LSI->IntroducerRange = IntroducerRange;
  LSI->ExplicitParams = ExplicitParams;
  LSI->Mutable = Mutable;

  if (ExplicitResultType) {
    LSI->ReturnType = CallOperator->getResultType();
    
    if (!LSI->ReturnType->isDependentType() &&
        !LSI->ReturnType->isVoidType()) {
      if (RequireCompleteType(CallOperator->getLocStart(), LSI->ReturnType,
                              diag::err_lambda_incomplete_result)) {
        // Do nothing.
      } else if (LSI->ReturnType->isObjCObjectOrInterfaceType()) {
        Diag(CallOperator->getLocStart(), diag::err_lambda_objc_object_result)
          << LSI->ReturnType;
      }
    }
  } else {
    LSI->HasImplicitReturnType = true;
  }

  return LSI;
}

void Sema::finishLambdaExplicitCaptures(LambdaScopeInfo *LSI) {
  LSI->finishedExplicitCaptures();
}

void Sema::addLambdaParameters(CXXMethodDecl *CallOperator, Scope *CurScope) {  
  // Introduce our parameters into the function scope
  for (unsigned p = 0, NumParams = CallOperator->getNumParams(); 
       p < NumParams; ++p) {
    ParmVarDecl *Param = CallOperator->getParamDecl(p);
    
    // If this has an identifier, add it to the scope stack.
    if (CurScope && Param->getIdentifier()) {
      CheckShadow(CurScope, Param);
      
      PushOnScopeChains(Param, CurScope);
    }
  }
}

static bool checkReturnValueType(const ASTContext &Ctx, const Expr *E,
                                 QualType &DeducedType,
                                 QualType &AlternateType) {
  // Handle ReturnStmts with no expressions.
  if (!E) {
    if (AlternateType.isNull())
      AlternateType = Ctx.VoidTy;

    return Ctx.hasSameType(DeducedType, Ctx.VoidTy);
  }

  QualType StrictType = E->getType();
  QualType LooseType = StrictType;

  // In C, enum constants have the type of their underlying integer type,
  // not the enum. When inferring block return types, we should allow
  // the enum type if an enum constant is used, unless the enum is
  // anonymous (in which case there can be no variables of its type).
  if (!Ctx.getLangOpts().CPlusPlus) {
    const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts());
    if (DRE) {
      const Decl *D = DRE->getDecl();
      if (const EnumConstantDecl *ECD = dyn_cast<EnumConstantDecl>(D)) {
        const EnumDecl *Enum = cast<EnumDecl>(ECD->getDeclContext());
        if (Enum->getDeclName() || Enum->getTypedefNameForAnonDecl())
          LooseType = Ctx.getTypeDeclType(Enum);
      }
    }
  }

  // Special case for the first return statement we find.
  // The return type has already been tentatively set, but we might still
  // have an alternate type we should prefer.
  if (AlternateType.isNull())
    AlternateType = LooseType;

  if (Ctx.hasSameType(DeducedType, StrictType)) {
    // FIXME: The loose type is different when there are constants from two
    // different enums. We could consider warning here.
    if (AlternateType != Ctx.DependentTy)
      if (!Ctx.hasSameType(AlternateType, LooseType))
        AlternateType = Ctx.VoidTy;
    return true;
  }

  if (Ctx.hasSameType(DeducedType, LooseType)) {
    // Use DependentTy to signal that we're using an alternate type and may
    // need to add casts somewhere.
    AlternateType = Ctx.DependentTy;
    return true;
  }

  if (Ctx.hasSameType(AlternateType, StrictType) ||
      Ctx.hasSameType(AlternateType, LooseType)) {
    DeducedType = AlternateType;
    // Use DependentTy to signal that we're using an alternate type and may
    // need to add casts somewhere.
    AlternateType = Ctx.DependentTy;
    return true;
  }

  return false;
}

void Sema::deduceClosureReturnType(CapturingScopeInfo &CSI) {
  assert(CSI.HasImplicitReturnType);

  // First case: no return statements, implicit void return type.
  ASTContext &Ctx = getASTContext();
  if (CSI.Returns.empty()) {
    // It's possible there were simply no /valid/ return statements.
    // In this case, the first one we found may have at least given us a type.
    if (CSI.ReturnType.isNull())
      CSI.ReturnType = Ctx.VoidTy;
    return;
  }

  // Second case: at least one return statement has dependent type.
  // Delay type checking until instantiation.
  assert(!CSI.ReturnType.isNull() && "We should have a tentative return type.");
  if (CSI.ReturnType->isDependentType())
    return;

  // Third case: only one return statement. Don't bother doing extra work!
  SmallVectorImpl<ReturnStmt*>::iterator I = CSI.Returns.begin(),
                                         E = CSI.Returns.end();
  if (I+1 == E)
    return;

  // General case: many return statements.
  // Check that they all have compatible return types.
  // For now, that means "identical", with an exception for enum constants.
  // (In C, enum constants have the type of their underlying integer type,
  // not the type of the enum. C++ uses the type of the enum.)
  QualType AlternateType;

  // We require the return types to strictly match here.
  for (; I != E; ++I) {
    const ReturnStmt *RS = *I;
    const Expr *RetE = RS->getRetValue();
    if (!checkReturnValueType(Ctx, RetE, CSI.ReturnType, AlternateType)) {
      // FIXME: This is a poor diagnostic for ReturnStmts without expressions.
      Diag(RS->getLocStart(),
           diag::err_typecheck_missing_return_type_incompatible)
        << (RetE ? RetE->getType() : Ctx.VoidTy) << CSI.ReturnType
        << isa<LambdaScopeInfo>(CSI);
      // Don't bother fixing up the return statements in the block if some of
      // them are unfixable anyway.
      AlternateType = Ctx.VoidTy;
      // Continue iterating so that we keep emitting diagnostics.
    }
  }

  // If our return statements turned out to be compatible, but we needed to
  // pick a different return type, go through and fix the ones that need it.
  if (AlternateType == Ctx.DependentTy) {
    for (SmallVectorImpl<ReturnStmt*>::iterator I = CSI.Returns.begin(),
                                                E = CSI.Returns.end();
         I != E; ++I) {
      ReturnStmt *RS = *I;
      Expr *RetE = RS->getRetValue();
      if (RetE->getType() == CSI.ReturnType)
        continue;

      // Right now we only support integral fixup casts.
      assert(CSI.ReturnType->isIntegralOrUnscopedEnumerationType());
      assert(RetE->getType()->isIntegralOrUnscopedEnumerationType());
      ExprResult Casted = ImpCastExprToType(RetE, CSI.ReturnType,
                                            CK_IntegralCast);
      assert(Casted.isUsable());
      RS->setRetValue(Casted.take());
    }
  }
}

void Sema::ActOnStartOfLambdaDefinition(LambdaIntroducer &Intro,
                                        Declarator &ParamInfo,
                                        Scope *CurScope) {
  // Determine if we're within a context where we know that the lambda will
  // be dependent, because there are template parameters in scope.
  bool KnownDependent = false;
  if (Scope *TmplScope = CurScope->getTemplateParamParent())
    if (!TmplScope->decl_empty())
      KnownDependent = true;
  
  // Determine the signature of the call operator.
  TypeSourceInfo *MethodTyInfo;
  bool ExplicitParams = true;
  bool ExplicitResultType = true;
  bool ContainsUnexpandedParameterPack = false;
  SourceLocation EndLoc;
  llvm::SmallVector<ParmVarDecl *, 8> Params;
  if (ParamInfo.getNumTypeObjects() == 0) {
    // C++11 [expr.prim.lambda]p4:
    //   If a lambda-expression does not include a lambda-declarator, it is as 
    //   if the lambda-declarator were ().
    FunctionProtoType::ExtProtoInfo EPI;
    EPI.HasTrailingReturn = true;
    EPI.TypeQuals |= DeclSpec::TQ_const;
    QualType MethodTy = Context.getFunctionType(Context.DependentTy,
                                                /*Args=*/0, /*NumArgs=*/0, EPI);
    MethodTyInfo = Context.getTrivialTypeSourceInfo(MethodTy);
    ExplicitParams = false;
    ExplicitResultType = false;
    EndLoc = Intro.Range.getEnd();
  } else {
    assert(ParamInfo.isFunctionDeclarator() &&
           "lambda-declarator is a function");
    DeclaratorChunk::FunctionTypeInfo &FTI = ParamInfo.getFunctionTypeInfo();
    
    // C++11 [expr.prim.lambda]p5:
    //   This function call operator is declared const (9.3.1) if and only if 
    //   the lambda-expression's parameter-declaration-clause is not followed 
    //   by mutable. It is neither virtual nor declared volatile. [...]
    if (!FTI.hasMutableQualifier())
      FTI.TypeQuals |= DeclSpec::TQ_const;
    
    MethodTyInfo = GetTypeForDeclarator(ParamInfo, CurScope);
    assert(MethodTyInfo && "no type from lambda-declarator");
    EndLoc = ParamInfo.getSourceRange().getEnd();
    
    ExplicitResultType
      = MethodTyInfo->getType()->getAs<FunctionType>()->getResultType() 
                                                        != Context.DependentTy;

    if (FTI.NumArgs == 1 && !FTI.isVariadic && FTI.ArgInfo[0].Ident == 0 &&
        cast<ParmVarDecl>(FTI.ArgInfo[0].Param)->getType()->isVoidType()) {
      // Empty arg list, don't push any params.
      checkVoidParamDecl(cast<ParmVarDecl>(FTI.ArgInfo[0].Param));
    } else {
      Params.reserve(FTI.NumArgs);
      for (unsigned i = 0, e = FTI.NumArgs; i != e; ++i)
        Params.push_back(cast<ParmVarDecl>(FTI.ArgInfo[i].Param));
    }

    // Check for unexpanded parameter packs in the method type.
    if (MethodTyInfo->getType()->containsUnexpandedParameterPack())
      ContainsUnexpandedParameterPack = true;
  }

  CXXRecordDecl *Class = createLambdaClosureType(Intro.Range, MethodTyInfo,
                                                 KnownDependent);

  CXXMethodDecl *Method = startLambdaDefinition(Class, Intro.Range,
                                                MethodTyInfo, EndLoc, Params);
  
  if (ExplicitParams)
    CheckCXXDefaultArguments(Method);
  
  // Attributes on the lambda apply to the method.  
  ProcessDeclAttributes(CurScope, Method, ParamInfo);
  
  // Introduce the function call operator as the current declaration context.
  PushDeclContext(CurScope, Method);
    
  // Introduce the lambda scope.
  LambdaScopeInfo *LSI
    = enterLambdaScope(Method, Intro.Range, Intro.Default, ExplicitParams,
                       ExplicitResultType,
                       !Method->isConst());
 
  // Handle explicit captures.
  SourceLocation PrevCaptureLoc
    = Intro.Default == LCD_None? Intro.Range.getBegin() : Intro.DefaultLoc;
  for (llvm::SmallVector<LambdaCapture, 4>::const_iterator
         C = Intro.Captures.begin(), 
         E = Intro.Captures.end(); 
       C != E; 
       PrevCaptureLoc = C->Loc, ++C) {
    if (C->Kind == LCK_This) {
      // C++11 [expr.prim.lambda]p8:
      //   An identifier or this shall not appear more than once in a 
      //   lambda-capture.
      if (LSI->isCXXThisCaptured()) {
        Diag(C->Loc, diag::err_capture_more_than_once) 
          << "'this'"
          << SourceRange(LSI->getCXXThisCapture().getLocation())
          << FixItHint::CreateRemoval(
               SourceRange(PP.getLocForEndOfToken(PrevCaptureLoc), C->Loc));
        continue;
      }

      // C++11 [expr.prim.lambda]p8:
      //   If a lambda-capture includes a capture-default that is =, the 
      //   lambda-capture shall not contain this [...].
      if (Intro.Default == LCD_ByCopy) {
        Diag(C->Loc, diag::err_this_capture_with_copy_default)
          << FixItHint::CreateRemoval(
               SourceRange(PP.getLocForEndOfToken(PrevCaptureLoc), C->Loc));
        continue;
      }

      // C++11 [expr.prim.lambda]p12:
      //   If this is captured by a local lambda expression, its nearest
      //   enclosing function shall be a non-static member function.
      QualType ThisCaptureType = getCurrentThisType();
      if (ThisCaptureType.isNull()) {
        Diag(C->Loc, diag::err_this_capture) << true;
        continue;
      }
      
      CheckCXXThisCapture(C->Loc, /*Explicit=*/true);
      continue;
    }

    assert(C->Id && "missing identifier for capture");

    // C++11 [expr.prim.lambda]p8:
    //   If a lambda-capture includes a capture-default that is &, the 
    //   identifiers in the lambda-capture shall not be preceded by &.
    //   If a lambda-capture includes a capture-default that is =, [...]
    //   each identifier it contains shall be preceded by &.
    if (C->Kind == LCK_ByRef && Intro.Default == LCD_ByRef) {
      Diag(C->Loc, diag::err_reference_capture_with_reference_default)
        << FixItHint::CreateRemoval(
             SourceRange(PP.getLocForEndOfToken(PrevCaptureLoc), C->Loc));
      continue;
    } else if (C->Kind == LCK_ByCopy && Intro.Default == LCD_ByCopy) {
      Diag(C->Loc, diag::err_copy_capture_with_copy_default)
        << FixItHint::CreateRemoval(
             SourceRange(PP.getLocForEndOfToken(PrevCaptureLoc), C->Loc));
      continue;
    }

    DeclarationNameInfo Name(C->Id, C->Loc);
    LookupResult R(*this, Name, LookupOrdinaryName);
    LookupName(R, CurScope);
    if (R.isAmbiguous())
      continue;
    if (R.empty()) {
      // FIXME: Disable corrections that would add qualification?
      CXXScopeSpec ScopeSpec;
      DeclFilterCCC<VarDecl> Validator;
      if (DiagnoseEmptyLookup(CurScope, ScopeSpec, R, Validator))
        continue;
    }

    // C++11 [expr.prim.lambda]p10:
    //   The identifiers in a capture-list are looked up using the usual rules
    //   for unqualified name lookup (3.4.1); each such lookup shall find a 
    //   variable with automatic storage duration declared in the reaching 
    //   scope of the local lambda expression.
    // 
    // Note that the 'reaching scope' check happens in tryCaptureVariable().
    VarDecl *Var = R.getAsSingle<VarDecl>();
    if (!Var) {
      Diag(C->Loc, diag::err_capture_does_not_name_variable) << C->Id;
      continue;
    }

    // Ignore invalid decls; they'll just confuse the code later.
    if (Var->isInvalidDecl())
      continue;

    if (!Var->hasLocalStorage()) {
      Diag(C->Loc, diag::err_capture_non_automatic_variable) << C->Id;
      Diag(Var->getLocation(), diag::note_previous_decl) << C->Id;
      continue;
    }

    // C++11 [expr.prim.lambda]p8:
    //   An identifier or this shall not appear more than once in a 
    //   lambda-capture.
    if (LSI->isCaptured(Var)) {
      Diag(C->Loc, diag::err_capture_more_than_once) 
        << C->Id
        << SourceRange(LSI->getCapture(Var).getLocation())
        << FixItHint::CreateRemoval(
             SourceRange(PP.getLocForEndOfToken(PrevCaptureLoc), C->Loc));
      continue;
    }

    // C++11 [expr.prim.lambda]p23:
    //   A capture followed by an ellipsis is a pack expansion (14.5.3).
    SourceLocation EllipsisLoc;
    if (C->EllipsisLoc.isValid()) {
      if (Var->isParameterPack()) {
        EllipsisLoc = C->EllipsisLoc;
      } else {
        Diag(C->EllipsisLoc, diag::err_pack_expansion_without_parameter_packs)
          << SourceRange(C->Loc);
        
        // Just ignore the ellipsis.
      }
    } else if (Var->isParameterPack()) {
      ContainsUnexpandedParameterPack = true;
    }
    
    TryCaptureKind Kind = C->Kind == LCK_ByRef ? TryCapture_ExplicitByRef :
                                                 TryCapture_ExplicitByVal;
    tryCaptureVariable(Var, C->Loc, Kind, EllipsisLoc);
  }
  finishLambdaExplicitCaptures(LSI);

  LSI->ContainsUnexpandedParameterPack = ContainsUnexpandedParameterPack;

  // Add lambda parameters into scope.
  addLambdaParameters(Method, CurScope);

  // Enter a new evaluation context to insulate the lambda from any
  // cleanups from the enclosing full-expression.
  PushExpressionEvaluationContext(PotentiallyEvaluated);  
}

void Sema::ActOnLambdaError(SourceLocation StartLoc, Scope *CurScope,
                            bool IsInstantiation) {
  // Leave the expression-evaluation context.
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();

  // Leave the context of the lambda.
  if (!IsInstantiation)
    PopDeclContext();

  // Finalize the lambda.
  LambdaScopeInfo *LSI = getCurLambda();
  CXXRecordDecl *Class = LSI->Lambda;
  Class->setInvalidDecl();
  SmallVector<Decl*, 4> Fields;
  for (RecordDecl::field_iterator i = Class->field_begin(),
                                  e = Class->field_end(); i != e; ++i)
    Fields.push_back(*i);
  ActOnFields(0, Class->getLocation(), Class, Fields, 
              SourceLocation(), SourceLocation(), 0);
  CheckCompletedCXXClass(Class);

  PopFunctionScopeInfo();
}

/// \brief Add a lambda's conversion to function pointer, as described in
/// C++11 [expr.prim.lambda]p6.
static void addFunctionPointerConversion(Sema &S, 
                                         SourceRange IntroducerRange,
                                         CXXRecordDecl *Class,
                                         CXXMethodDecl *CallOperator) {
  // Add the conversion to function pointer.
  const FunctionProtoType *Proto
    = CallOperator->getType()->getAs<FunctionProtoType>(); 
  QualType FunctionPtrTy;
  QualType FunctionTy;
  {
    FunctionProtoType::ExtProtoInfo ExtInfo = Proto->getExtProtoInfo();
    ExtInfo.TypeQuals = 0;
    FunctionTy = S.Context.getFunctionType(Proto->getResultType(),
                                           Proto->arg_type_begin(),
                                           Proto->getNumArgs(),
                                           ExtInfo);
    FunctionPtrTy = S.Context.getPointerType(FunctionTy);
  }
  
  FunctionProtoType::ExtProtoInfo ExtInfo;
  ExtInfo.TypeQuals = Qualifiers::Const;
  QualType ConvTy = S.Context.getFunctionType(FunctionPtrTy, 0, 0, ExtInfo);
  
  SourceLocation Loc = IntroducerRange.getBegin();
  DeclarationName Name
    = S.Context.DeclarationNames.getCXXConversionFunctionName(
        S.Context.getCanonicalType(FunctionPtrTy));
  DeclarationNameLoc NameLoc;
  NameLoc.NamedType.TInfo = S.Context.getTrivialTypeSourceInfo(FunctionPtrTy,
                                                               Loc);
  CXXConversionDecl *Conversion 
    = CXXConversionDecl::Create(S.Context, Class, Loc, 
                                DeclarationNameInfo(Name, Loc, NameLoc),
                                ConvTy, 
                                S.Context.getTrivialTypeSourceInfo(ConvTy, 
                                                                   Loc),
                                /*isInline=*/false, /*isExplicit=*/false,
                                /*isConstexpr=*/false, 
                                CallOperator->getBody()->getLocEnd());
  Conversion->setAccess(AS_public);
  Conversion->setImplicit(true);
  Class->addDecl(Conversion);
  
  // Add a non-static member function "__invoke" that will be the result of
  // the conversion.
  Name = &S.Context.Idents.get("__invoke");
  CXXMethodDecl *Invoke
    = CXXMethodDecl::Create(S.Context, Class, Loc, 
                            DeclarationNameInfo(Name, Loc), FunctionTy, 
                            CallOperator->getTypeSourceInfo(),
                            /*IsStatic=*/true, SC_Static, /*IsInline=*/true,
                            /*IsConstexpr=*/false, 
                            CallOperator->getBody()->getLocEnd());
  SmallVector<ParmVarDecl *, 4> InvokeParams;
  for (unsigned I = 0, N = CallOperator->getNumParams(); I != N; ++I) {
    ParmVarDecl *From = CallOperator->getParamDecl(I);
    InvokeParams.push_back(ParmVarDecl::Create(S.Context, Invoke,
                                               From->getLocStart(),
                                               From->getLocation(),
                                               From->getIdentifier(),
                                               From->getType(),
                                               From->getTypeSourceInfo(),
                                               From->getStorageClass(),
                                               From->getStorageClassAsWritten(),
                                               /*DefaultArg=*/0));
  }
  Invoke->setParams(InvokeParams);
  Invoke->setAccess(AS_private);
  Invoke->setImplicit(true);
  Class->addDecl(Invoke);
}

/// \brief Add a lambda's conversion to block pointer.
static void addBlockPointerConversion(Sema &S, 
                                      SourceRange IntroducerRange,
                                      CXXRecordDecl *Class,
                                      CXXMethodDecl *CallOperator) {
  const FunctionProtoType *Proto
    = CallOperator->getType()->getAs<FunctionProtoType>(); 
  QualType BlockPtrTy;
  {
    FunctionProtoType::ExtProtoInfo ExtInfo = Proto->getExtProtoInfo();
    ExtInfo.TypeQuals = 0;
    QualType FunctionTy
      = S.Context.getFunctionType(Proto->getResultType(),
                                  Proto->arg_type_begin(),
                                  Proto->getNumArgs(),
                                  ExtInfo);
    BlockPtrTy = S.Context.getBlockPointerType(FunctionTy);
  }
  
  FunctionProtoType::ExtProtoInfo ExtInfo;
  ExtInfo.TypeQuals = Qualifiers::Const;
  QualType ConvTy = S.Context.getFunctionType(BlockPtrTy, 0, 0, ExtInfo);
  
  SourceLocation Loc = IntroducerRange.getBegin();
  DeclarationName Name
    = S.Context.DeclarationNames.getCXXConversionFunctionName(
        S.Context.getCanonicalType(BlockPtrTy));
  DeclarationNameLoc NameLoc;
  NameLoc.NamedType.TInfo = S.Context.getTrivialTypeSourceInfo(BlockPtrTy, Loc);
  CXXConversionDecl *Conversion 
    = CXXConversionDecl::Create(S.Context, Class, Loc, 
                                DeclarationNameInfo(Name, Loc, NameLoc),
                                ConvTy, 
                                S.Context.getTrivialTypeSourceInfo(ConvTy, Loc),
                                /*isInline=*/false, /*isExplicit=*/false,
                                /*isConstexpr=*/false, 
                                CallOperator->getBody()->getLocEnd());
  Conversion->setAccess(AS_public);
  Conversion->setImplicit(true);
  Class->addDecl(Conversion);
}
         
ExprResult Sema::ActOnLambdaExpr(SourceLocation StartLoc, Stmt *Body, 
                                 Scope *CurScope, 
                                 bool IsInstantiation) {
  // Collect information from the lambda scope.
  llvm::SmallVector<LambdaExpr::Capture, 4> Captures;
  llvm::SmallVector<Expr *, 4> CaptureInits;
  LambdaCaptureDefault CaptureDefault;
  CXXRecordDecl *Class;
  CXXMethodDecl *CallOperator;
  SourceRange IntroducerRange;
  bool ExplicitParams;
  bool ExplicitResultType;
  bool LambdaExprNeedsCleanups;
  bool ContainsUnexpandedParameterPack;
  llvm::SmallVector<VarDecl *, 4> ArrayIndexVars;
  llvm::SmallVector<unsigned, 4> ArrayIndexStarts;
  {
    LambdaScopeInfo *LSI = getCurLambda();
    CallOperator = LSI->CallOperator;
    Class = LSI->Lambda;
    IntroducerRange = LSI->IntroducerRange;
    ExplicitParams = LSI->ExplicitParams;
    ExplicitResultType = !LSI->HasImplicitReturnType;
    LambdaExprNeedsCleanups = LSI->ExprNeedsCleanups;
    ContainsUnexpandedParameterPack = LSI->ContainsUnexpandedParameterPack;
    ArrayIndexVars.swap(LSI->ArrayIndexVars);
    ArrayIndexStarts.swap(LSI->ArrayIndexStarts);
    
    // Translate captures.
    for (unsigned I = 0, N = LSI->Captures.size(); I != N; ++I) {
      LambdaScopeInfo::Capture From = LSI->Captures[I];
      assert(!From.isBlockCapture() && "Cannot capture __block variables");
      bool IsImplicit = I >= LSI->NumExplicitCaptures;

      // Handle 'this' capture.
      if (From.isThisCapture()) {
        Captures.push_back(LambdaExpr::Capture(From.getLocation(),
                                               IsImplicit,
                                               LCK_This));
        CaptureInits.push_back(new (Context) CXXThisExpr(From.getLocation(),
                                                         getCurrentThisType(),
                                                         /*isImplicit=*/true));
        continue;
      }

      VarDecl *Var = From.getVariable();
      LambdaCaptureKind Kind = From.isCopyCapture()? LCK_ByCopy : LCK_ByRef;
      Captures.push_back(LambdaExpr::Capture(From.getLocation(), IsImplicit, 
                                             Kind, Var, From.getEllipsisLoc()));
      CaptureInits.push_back(From.getCopyExpr());
    }

    switch (LSI->ImpCaptureStyle) {
    case CapturingScopeInfo::ImpCap_None:
      CaptureDefault = LCD_None;
      break;

    case CapturingScopeInfo::ImpCap_LambdaByval:
      CaptureDefault = LCD_ByCopy;
      break;

    case CapturingScopeInfo::ImpCap_LambdaByref:
      CaptureDefault = LCD_ByRef;
      break;

    case CapturingScopeInfo::ImpCap_Block:
      llvm_unreachable("block capture in lambda");
      break;
    }

    // C++11 [expr.prim.lambda]p4:
    //   If a lambda-expression does not include a
    //   trailing-return-type, it is as if the trailing-return-type
    //   denotes the following type:
    // FIXME: Assumes current resolution to core issue 975.
    if (LSI->HasImplicitReturnType) {
      deduceClosureReturnType(*LSI);

      //   - if there are no return statements in the
      //     compound-statement, or all return statements return
      //     either an expression of type void or no expression or
      //     braced-init-list, the type void;
      if (LSI->ReturnType.isNull()) {
        LSI->ReturnType = Context.VoidTy;
      }

      // Create a function type with the inferred return type.
      const FunctionProtoType *Proto
        = CallOperator->getType()->getAs<FunctionProtoType>();
      QualType FunctionTy
        = Context.getFunctionType(LSI->ReturnType,
                                  Proto->arg_type_begin(),
                                  Proto->getNumArgs(),
                                  Proto->getExtProtoInfo());
      CallOperator->setType(FunctionTy);
    }

    // C++ [expr.prim.lambda]p7:
    //   The lambda-expression's compound-statement yields the
    //   function-body (8.4) of the function call operator [...].
    ActOnFinishFunctionBody(CallOperator, Body, IsInstantiation);
    CallOperator->setLexicalDeclContext(Class);
    Class->addDecl(CallOperator);
    PopExpressionEvaluationContext();

    // C++11 [expr.prim.lambda]p6:
    //   The closure type for a lambda-expression with no lambda-capture
    //   has a public non-virtual non-explicit const conversion function
    //   to pointer to function having the same parameter and return
    //   types as the closure type's function call operator.
    if (Captures.empty() && CaptureDefault == LCD_None)
      addFunctionPointerConversion(*this, IntroducerRange, Class,
                                   CallOperator);

    // Objective-C++:
    //   The closure type for a lambda-expression has a public non-virtual
    //   non-explicit const conversion function to a block pointer having the
    //   same parameter and return types as the closure type's function call
    //   operator.
    if (getLangOpts().Blocks && getLangOpts().ObjC1)
      addBlockPointerConversion(*this, IntroducerRange, Class, CallOperator);
    
    // Finalize the lambda class.
    SmallVector<Decl*, 4> Fields;
    for (RecordDecl::field_iterator i = Class->field_begin(),
                                    e = Class->field_end(); i != e; ++i)
      Fields.push_back(*i);
    ActOnFields(0, Class->getLocation(), Class, Fields, 
                SourceLocation(), SourceLocation(), 0);
    CheckCompletedCXXClass(Class);
  }

  if (LambdaExprNeedsCleanups)
    ExprNeedsCleanups = true;
  
  LambdaExpr *Lambda = LambdaExpr::Create(Context, Class, IntroducerRange, 
                                          CaptureDefault, Captures, 
                                          ExplicitParams, ExplicitResultType,
                                          CaptureInits, ArrayIndexVars, 
                                          ArrayIndexStarts, Body->getLocEnd(),
                                          ContainsUnexpandedParameterPack);

  // C++11 [expr.prim.lambda]p2:
  //   A lambda-expression shall not appear in an unevaluated operand
  //   (Clause 5).
  if (!CurContext->isDependentContext()) {
    switch (ExprEvalContexts.back().Context) {
    case Unevaluated:
      // We don't actually diagnose this case immediately, because we
      // could be within a context where we might find out later that
      // the expression is potentially evaluated (e.g., for typeid).
      ExprEvalContexts.back().Lambdas.push_back(Lambda);
      break;

    case ConstantEvaluated:
    case PotentiallyEvaluated:
    case PotentiallyEvaluatedIfUsed:
      break;
    }
  }
  
  return MaybeBindToTemporary(Lambda);
}

ExprResult Sema::BuildBlockForLambdaConversion(SourceLocation CurrentLocation,
                                               SourceLocation ConvLocation,
                                               CXXConversionDecl *Conv,
                                               Expr *Src) {
  // Make sure that the lambda call operator is marked used.
  CXXRecordDecl *Lambda = Conv->getParent();
  CXXMethodDecl *CallOperator 
    = cast<CXXMethodDecl>(
        *Lambda->lookup(
          Context.DeclarationNames.getCXXOperatorName(OO_Call)).first);
  CallOperator->setReferenced();
  CallOperator->setUsed();

  ExprResult Init = PerformCopyInitialization(
                      InitializedEntity::InitializeBlock(ConvLocation, 
                                                         Src->getType(), 
                                                         /*NRVO=*/false),
                      CurrentLocation, Src);
  if (!Init.isInvalid())
    Init = ActOnFinishFullExpr(Init.take());
  
  if (Init.isInvalid())
    return ExprError();
  
  // Create the new block to be returned.
  BlockDecl *Block = BlockDecl::Create(Context, CurContext, ConvLocation);

  // Set the type information.
  Block->setSignatureAsWritten(CallOperator->getTypeSourceInfo());
  Block->setIsVariadic(CallOperator->isVariadic());
  Block->setBlockMissingReturnType(false);

  // Add parameters.
  SmallVector<ParmVarDecl *, 4> BlockParams;
  for (unsigned I = 0, N = CallOperator->getNumParams(); I != N; ++I) {
    ParmVarDecl *From = CallOperator->getParamDecl(I);
    BlockParams.push_back(ParmVarDecl::Create(Context, Block,
                                              From->getLocStart(),
                                              From->getLocation(),
                                              From->getIdentifier(),
                                              From->getType(),
                                              From->getTypeSourceInfo(),
                                              From->getStorageClass(),
                                            From->getStorageClassAsWritten(),
                                              /*DefaultArg=*/0));
  }
  Block->setParams(BlockParams);

  Block->setIsConversionFromLambda(true);

  // Add capture. The capture uses a fake variable, which doesn't correspond
  // to any actual memory location. However, the initializer copy-initializes
  // the lambda object.
  TypeSourceInfo *CapVarTSI =
      Context.getTrivialTypeSourceInfo(Src->getType());
  VarDecl *CapVar = VarDecl::Create(Context, Block, ConvLocation,
                                    ConvLocation, 0,
                                    Src->getType(), CapVarTSI,
                                    SC_None, SC_None);
  BlockDecl::Capture Capture(/*Variable=*/CapVar, /*ByRef=*/false,
                             /*Nested=*/false, /*Copy=*/Init.take());
  Block->setCaptures(Context, &Capture, &Capture + 1, 
                     /*CapturesCXXThis=*/false);

  // Add a fake function body to the block. IR generation is responsible
  // for filling in the actual body, which cannot be expressed as an AST.
  Block->setBody(new (Context) CompoundStmt(ConvLocation));

  // Create the block literal expression.
  Expr *BuildBlock = new (Context) BlockExpr(Block, Conv->getConversionType());
  ExprCleanupObjects.push_back(Block);
  ExprNeedsCleanups = true;

  return BuildBlock;
}
