//===- SemaSYCL.cpp - Semantic Analysis for SYCL constructs ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for SYCL constructs.
//===----------------------------------------------------------------------===//

#include "clang/AST/AST.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;

LambdaExpr *getBodyAsLambda(CXXMemberCallExpr *e) {
  auto LastArg = e->getArg(e->getNumArgs() - 1);
  return dyn_cast<LambdaExpr>(LastArg);
}

FunctionDecl *CreateSYCLKernelFunction(ASTContext &Context, StringRef Name,
                                       ArrayRef<QualType> ArgTys,
                                       ArrayRef<DeclaratorDecl *> ArgDecls) {

  DeclContext *DC = Context.getTranslationUnitDecl();
  FunctionProtoType::ExtProtoInfo Info(CC_OpenCLKernel);
  QualType RetTy = Context.VoidTy;
  QualType FuncTy = Context.getFunctionType(RetTy, ArgTys, Info);
  DeclarationName DN = DeclarationName(&Context.Idents.get(Name));
  FunctionDecl *Result = FunctionDecl::Create(
      Context, DC, SourceLocation(), SourceLocation(), DN, FuncTy,
      Context.getTrivialTypeSourceInfo(RetTy), SC_None);
  llvm::SmallVector<ParmVarDecl *, 16> Params;
  int i = 0;
  for (auto ArgTy : ArgTys) {
    auto P =
        ParmVarDecl::Create(Context, Result, SourceLocation(), SourceLocation(),
                            ArgDecls[i]->getIdentifier(), ArgTy,
                            ArgDecls[i]->getTypeSourceInfo(), SC_None, 0);
    P->setScopeInfo(0, i++);
    P->setIsUsed();
    Params.push_back(P);
  }
  Result->setParams(Params);
  // TODO: Add SYCL specific attribute for kernel and all functions called
  // by kernel.
  Result->addAttr(OpenCLKernelAttr::CreateImplicit(Context));
  Result->addAttr(AsmLabelAttr::CreateImplicit(Context, Name));
  return Result;
}

CompoundStmt *CreateSYCLKernelBody(Sema &S, CXXMemberCallExpr *e,
                                   DeclContext *DC) {

  llvm::SmallVector<Stmt *, 16> BodyStmts;

  // TODO: case when kernel is functor
  // TODO: possible refactoring when functor case will be completed
  LambdaExpr *LE = getBodyAsLambda(e);
  if (LE) {
    // Create Lambda object
    CXXRecordDecl *LC = LE->getLambdaClass();
    auto LambdaVD = VarDecl::Create(
        S.Context, DC, SourceLocation(), SourceLocation(), LC->getIdentifier(),
        QualType(LC->getTypeForDecl(), 0), LC->getLambdaTypeInfo(), SC_None);

    Stmt *DS = new (S.Context)
        DeclStmt(DeclGroupRef(LambdaVD), SourceLocation(), SourceLocation());
    BodyStmts.push_back(DS);
    auto LambdaDRE = DeclRefExpr::Create(
        S.Context, NestedNameSpecifierLoc(), SourceLocation(), LambdaVD, false,
        DeclarationNameInfo(), QualType(LC->getTypeForDecl(), 0), VK_LValue);

    // Init Lambda fields
    llvm::SmallVector<Expr *, 16> InitCaptures;

    auto TargetFunc = dyn_cast<FunctionDecl>(DC);
    auto TargetFuncParam =
        TargetFunc->param_begin(); // Iterator to ParamVarDecl (VarDecl)
    for (auto Field : LC->fields()) {
      QualType ParamType = (*TargetFuncParam)->getOriginalType();
      auto DRE = DeclRefExpr::Create(
          S.Context, NestedNameSpecifierLoc(), SourceLocation(),
          *TargetFuncParam, false, DeclarationNameInfo(), ParamType, VK_LValue);

      CXXRecordDecl *CRD = Field->getType()->getAsCXXRecordDecl();
      if (CRD) {
        llvm::SmallVector<Expr *, 16> ParamStmts;
        DeclAccessPair FieldDAP = DeclAccessPair::make(Field, AS_none);
        auto AccessorME = MemberExpr::Create(
            S.Context, LambdaDRE, false, SourceLocation(),
            NestedNameSpecifierLoc(), SourceLocation(), Field, FieldDAP,
            DeclarationNameInfo(Field->getDeclName(), SourceLocation()),
            nullptr, Field->getType(), VK_LValue, OK_Ordinary);

        for (auto Method : CRD->methods()) {
          if (Method->getNameInfo().getName().getAsString() ==
              "__set_pointer") {
            DeclAccessPair MethodDAP = DeclAccessPair::make(Method, AS_none);
            auto ME = MemberExpr::Create(
                S.Context, AccessorME, false, SourceLocation(),
                NestedNameSpecifierLoc(), SourceLocation(), Method, MethodDAP,
                Method->getNameInfo(), nullptr, Method->getType(), VK_LValue,
                OK_Ordinary);

            // Not referenced -> not emitted
            S.MarkFunctionReferenced(SourceLocation(), Method, true);

            QualType ResultTy = Method->getReturnType();
            ExprValueKind VK = Expr::getValueKindForType(ResultTy);
            ResultTy = ResultTy.getNonLValueExprType(S.Context);

            // __set_pointer needs one parameter
            QualType paramTy = (*(Method->param_begin()))->getOriginalType();

            // C++ address space attribute != opencl address space attribute
            Expr *qualifiersCast = ImplicitCastExpr::Create(
                S.Context, paramTy, CK_NoOp, DRE, nullptr, VK_LValue);
            Expr *Res =
                ImplicitCastExpr::Create(S.Context, paramTy, CK_LValueToRValue,
                                         qualifiersCast, nullptr, VK_RValue);

            ParamStmts.push_back(Res);

            // lambda.accessor.__set_pointer(kernel_parameter)
            CXXMemberCallExpr *Call = CXXMemberCallExpr::Create(
                S.Context, ME, ParamStmts, ResultTy, VK, SourceLocation());
            BodyStmts.push_back(Call);
          }
        }
      }
      TargetFuncParam++;
    }

    // Create Lambda operator () call
    FunctionDecl *LO = LE->getCallOperator();
    ArrayRef<ParmVarDecl *> Args = LO->parameters();
    llvm::SmallVector<Expr *, 16> ParamStmts(1);
    ParamStmts[0] = dyn_cast<Expr>(LambdaDRE);

    // Collect arguments for () operator
    for (auto Arg : Args) {
      QualType ArgType = Arg->getOriginalType();
      // Declare variable for parameter and pass it to call
      auto param_VD =
          VarDecl::Create(S.Context, DC, SourceLocation(), SourceLocation(),
                          Arg->getIdentifier(), ArgType,
                          S.Context.getTrivialTypeSourceInfo(ArgType), SC_None);
      Stmt *param_DS = new (S.Context)
          DeclStmt(DeclGroupRef(param_VD), SourceLocation(), SourceLocation());
      BodyStmts.push_back(param_DS);
      auto DRE = DeclRefExpr::Create(S.Context, NestedNameSpecifierLoc(),
                                     SourceLocation(), param_VD, false,
                                     DeclarationNameInfo(), ArgType, VK_LValue);
      Expr *Res = ImplicitCastExpr::Create(
          S.Context, ArgType, CK_LValueToRValue, DRE, nullptr, VK_RValue);
      ParamStmts.push_back(Res);
    }

    // Create ref for call operator
    DeclRefExpr *DRE = new (S.Context)
        DeclRefExpr(S.Context, LO, false, LO->getType(), VK_LValue,
                    SourceLocation());
    QualType ResultTy = LO->getReturnType();
    ExprValueKind VK = Expr::getValueKindForType(ResultTy);
    ResultTy = ResultTy.getNonLValueExprType(S.Context);

    CXXOperatorCallExpr *TheCall = CXXOperatorCallExpr::Create(
        S.Context, OO_Call, DRE, ParamStmts, ResultTy, VK, SourceLocation(), 
        FPOptions(), clang::CallExpr::ADLCallKind::NotADL );
    BodyStmts.push_back(TheCall);
  }
  return CompoundStmt::Create(S.Context, BodyStmts, SourceLocation(),
                              SourceLocation());
}

void BuildArgTys(ASTContext &Context,
                 llvm::SmallVector<DeclaratorDecl *, 16> &ArgDecls,
                 llvm::SmallVector<DeclaratorDecl *, 16> &NewArgDecls,
                 llvm::SmallVector<QualType, 16> &ArgTys) {
  for (auto V : ArgDecls) {
    QualType ArgTy = V->getType();
    QualType ActualArgType = ArgTy;
    StringRef Name = ArgTy.getBaseTypeIdentifier()->getName();
    // TODO: harden this check with additional validation that this class is
    // declared in cl::sycl namespace
    if (std::string(Name) == "accessor") {
      if (const auto *RecordDecl = ArgTy->getAsCXXRecordDecl()) {
        const auto *TemplateDecl =
            dyn_cast<ClassTemplateSpecializationDecl>(RecordDecl);
        if (TemplateDecl) {
          QualType PointeeType = TemplateDecl->getTemplateArgs()[0].getAsType();
          Qualifiers Quals = PointeeType.getQualifiers();
          // TODO: get address space from accessor template parameter.
          Quals.setAddressSpace(LangAS::opencl_global);
          PointeeType =
              Context.getQualifiedType(PointeeType.getUnqualifiedType(), Quals);
          QualType PointerType = Context.getPointerType(PointeeType);
          ActualArgType =
              Context.getQualifiedType(PointerType.getUnqualifiedType(), Quals);
        }
      }
    }
    DeclContext *DC = Context.getTranslationUnitDecl();

    IdentifierInfo *VarName = 0;
    SmallString<8> Str;
    llvm::raw_svector_ostream OS(Str);
    OS << "_arg_" << V->getIdentifier()->getName();
    VarName = &Context.Idents.get(OS.str());

    auto NewVarDecl = VarDecl::Create(
        Context, DC, SourceLocation(), SourceLocation(), VarName, ActualArgType,
        Context.getTrivialTypeSourceInfo(ActualArgType), SC_None);
    ArgTys.push_back(ActualArgType);
    NewArgDecls.push_back(NewVarDecl);
  }
}

void Sema::ConstructSYCLKernel(CXXMemberCallExpr *e) {
  // TODO: Case when kernel is functor
  LambdaExpr *LE = getBodyAsLambda(e);
  if (LE) {

    llvm::SmallVector<DeclaratorDecl *, 16> ArgDecls;

    for (const auto &V : LE->captures()) {
      ArgDecls.push_back(V.getCapturedVar());
    }

    llvm::SmallVector<QualType, 16> ArgTys;
    llvm::SmallVector<DeclaratorDecl *, 16> NewArgDecls;
    BuildArgTys(getASTContext(), ArgDecls, NewArgDecls, ArgTys);

    // Get Name for our kernel.
    FunctionDecl *FuncDecl = e->getMethodDecl();
    const TemplateArgumentList *TemplateArgs =
        FuncDecl->getTemplateSpecializationArgs();
    QualType KernelNameType = TemplateArgs->get(0).getAsType();
    std::string Name = KernelNameType.getBaseTypeIdentifier()->getName().str();

    if (const auto *RecordDecl = KernelNameType->getAsCXXRecordDecl()) {
      const auto *TemplateDecl =
          dyn_cast<ClassTemplateSpecializationDecl>(RecordDecl);
      if (TemplateDecl) {
        QualType ParamType = TemplateDecl->getTemplateArgs()[0].getAsType();
        Name += "_" + ParamType.getAsString() + "_";
      }
    }

    FunctionDecl *SYCLKernel =
        CreateSYCLKernelFunction(getASTContext(), Name, ArgTys, NewArgDecls);

    CompoundStmt *SYCLKernelBody = CreateSYCLKernelBody(*this, e, SYCLKernel);
    SYCLKernel->setBody(SYCLKernelBody);

    AddSyclKernel(SYCLKernel);
  }
}

