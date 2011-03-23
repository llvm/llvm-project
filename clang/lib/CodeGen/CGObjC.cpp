//===---- CGBuiltin.cpp - Emit LLVM Code for builtins ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Objective-C code as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGDebugInfo.h"
#include "CGObjCRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Target/TargetData.h"
using namespace clang;
using namespace CodeGen;

/// Emits an instance of NSConstantString representing the object.
llvm::Value *CodeGenFunction::EmitObjCStringLiteral(const ObjCStringLiteral *E)
{
  llvm::Constant *C = 
      CGM.getObjCRuntime().GenerateConstantString(E->getString());
  // FIXME: This bitcast should just be made an invariant on the Runtime.
  return llvm::ConstantExpr::getBitCast(C, ConvertType(E->getType()));
}

/// Emit a selector.
llvm::Value *CodeGenFunction::EmitObjCSelectorExpr(const ObjCSelectorExpr *E) {
  // Untyped selector.
  // Note that this implementation allows for non-constant strings to be passed
  // as arguments to @selector().  Currently, the only thing preventing this
  // behaviour is the type checking in the front end.
  return CGM.getObjCRuntime().GetSelector(Builder, E->getSelector());
}

llvm::Value *CodeGenFunction::EmitObjCProtocolExpr(const ObjCProtocolExpr *E) {
  // FIXME: This should pass the Decl not the name.
  return CGM.getObjCRuntime().GenerateProtocolRef(Builder, E->getProtocol());
}


RValue CodeGenFunction::EmitObjCMessageExpr(const ObjCMessageExpr *E,
                                            ReturnValueSlot Return) {
  // Only the lookup mechanism and first two arguments of the method
  // implementation vary between runtimes.  We can get the receiver and
  // arguments in generic code.

  CGObjCRuntime &Runtime = CGM.getObjCRuntime();
  bool isSuperMessage = false;
  bool isClassMessage = false;
  ObjCInterfaceDecl *OID = 0;
  // Find the receiver
  llvm::Value *Receiver = 0;
  switch (E->getReceiverKind()) {
  case ObjCMessageExpr::Instance:
    Receiver = EmitScalarExpr(E->getInstanceReceiver());
    break;

  case ObjCMessageExpr::Class: {
    const ObjCObjectType *ObjTy
      = E->getClassReceiver()->getAs<ObjCObjectType>();
    assert(ObjTy && "Invalid Objective-C class message send");
    OID = ObjTy->getInterface();
    assert(OID && "Invalid Objective-C class message send");
    Receiver = Runtime.GetClass(Builder, OID);
    isClassMessage = true;
    break;
  }

  case ObjCMessageExpr::SuperInstance:
    Receiver = LoadObjCSelf();
    isSuperMessage = true;
    break;

  case ObjCMessageExpr::SuperClass:
    Receiver = LoadObjCSelf();
    isSuperMessage = true;
    isClassMessage = true;
    break;
  }

  CallArgList Args;
  EmitCallArgs(Args, E->getMethodDecl(), E->arg_begin(), E->arg_end());

  QualType ResultType =
    E->getMethodDecl() ? E->getMethodDecl()->getResultType() : E->getType();

  if (isSuperMessage) {
    // super is only valid in an Objective-C method
    const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
    bool isCategoryImpl = isa<ObjCCategoryImplDecl>(OMD->getDeclContext());
    return Runtime.GenerateMessageSendSuper(*this, Return, ResultType,
                                            E->getSelector(),
                                            OMD->getClassInterface(),
                                            isCategoryImpl,
                                            Receiver,
                                            isClassMessage,
                                            Args,
                                            E->getMethodDecl());
  }

  return Runtime.GenerateMessageSend(*this, Return, ResultType,
                                     E->getSelector(),
                                     Receiver, Args, OID,
                                     E->getMethodDecl());
}

/// StartObjCMethod - Begin emission of an ObjCMethod. This generates
/// the LLVM function and sets the other context used by
/// CodeGenFunction.
void CodeGenFunction::StartObjCMethod(const ObjCMethodDecl *OMD,
                                      const ObjCContainerDecl *CD) {
  FunctionArgList Args;
  // Check if we should generate debug info for this method.
  if (CGM.getModuleDebugInfo() && !OMD->hasAttr<NoDebugAttr>())
    DebugInfo = CGM.getModuleDebugInfo();

  llvm::Function *Fn = CGM.getObjCRuntime().GenerateMethod(OMD, CD);

  const CGFunctionInfo &FI = CGM.getTypes().getFunctionInfo(OMD);
  CGM.SetInternalFunctionAttributes(OMD, Fn, FI);

  Args.push_back(std::make_pair(OMD->getSelfDecl(),
                                OMD->getSelfDecl()->getType()));
  Args.push_back(std::make_pair(OMD->getCmdDecl(),
                                OMD->getCmdDecl()->getType()));

  for (ObjCMethodDecl::param_iterator PI = OMD->param_begin(),
       E = OMD->param_end(); PI != E; ++PI)
    Args.push_back(std::make_pair(*PI, (*PI)->getType()));

  CurGD = OMD;

  StartFunction(OMD, OMD->getResultType(), Fn, Args, OMD->getLocStart());
}

void CodeGenFunction::GenerateObjCGetterBody(ObjCIvarDecl *Ivar, 
                                             bool IsAtomic, bool IsStrong) {
  LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), 
                                Ivar, 0);
  llvm::Value *GetCopyStructFn =
  CGM.getObjCRuntime().GetGetStructFunction();
  CodeGenTypes &Types = CGM.getTypes();
  // objc_copyStruct (ReturnValue, &structIvar, 
  //                  sizeof (Type of Ivar), isAtomic, false);
  CallArgList Args;
  RValue RV = RValue::get(Builder.CreateBitCast(ReturnValue,
                                                Types.ConvertType(getContext().VoidPtrTy)));
  Args.push_back(std::make_pair(RV, getContext().VoidPtrTy));
  RV = RValue::get(Builder.CreateBitCast(LV.getAddress(),
                                         Types.ConvertType(getContext().VoidPtrTy)));
  Args.push_back(std::make_pair(RV, getContext().VoidPtrTy));
  // sizeof (Type of Ivar)
  CharUnits Size =  getContext().getTypeSizeInChars(Ivar->getType());
  llvm::Value *SizeVal =
  llvm::ConstantInt::get(Types.ConvertType(getContext().LongTy), 
                         Size.getQuantity());
  Args.push_back(std::make_pair(RValue::get(SizeVal),
                                getContext().LongTy));
  llvm::Value *isAtomic =
  llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 
                         IsAtomic ? 1 : 0);
  Args.push_back(std::make_pair(RValue::get(isAtomic), 
                                getContext().BoolTy));
  llvm::Value *hasStrong =
  llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 
                         IsStrong ? 1 : 0);
  Args.push_back(std::make_pair(RValue::get(hasStrong), 
                                getContext().BoolTy));
  EmitCall(Types.getFunctionInfo(getContext().VoidTy, Args,
                                 FunctionType::ExtInfo()),
           GetCopyStructFn, ReturnValueSlot(), Args);
}

/// Generate an Objective-C method.  An Objective-C method is a C function with
/// its pointer, name, and types registered in the class struture.
void CodeGenFunction::GenerateObjCMethod(const ObjCMethodDecl *OMD) {
  StartObjCMethod(OMD, OMD->getClassInterface());
  EmitStmt(OMD->getBody());
  FinishFunction(OMD->getBodyRBrace());
}

// FIXME: I wasn't sure about the synthesis approach. If we end up generating an
// AST for the whole body we can just fall back to having a GenerateFunction
// which takes the body Stmt.

/// GenerateObjCGetter - Generate an Objective-C property getter
/// function. The given Decl must be an ObjCImplementationDecl. @synthesize
/// is illegal within a category.
void CodeGenFunction::GenerateObjCGetter(ObjCImplementationDecl *IMP,
                                         const ObjCPropertyImplDecl *PID) {
  ObjCIvarDecl *Ivar = PID->getPropertyIvarDecl();
  const ObjCPropertyDecl *PD = PID->getPropertyDecl();
  bool IsAtomic =
    !(PD->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_nonatomic);
  ObjCMethodDecl *OMD = PD->getGetterMethodDecl();
  assert(OMD && "Invalid call to generate getter (empty method)");
  StartObjCMethod(OMD, IMP->getClassInterface());
  
  // Determine if we should use an objc_getProperty call for
  // this. Non-atomic properties are directly evaluated.
  // atomic 'copy' and 'retain' properties are also directly
  // evaluated in gc-only mode.
  if (CGM.getLangOptions().getGCMode() != LangOptions::GCOnly &&
      IsAtomic &&
      (PD->getSetterKind() == ObjCPropertyDecl::Copy ||
       PD->getSetterKind() == ObjCPropertyDecl::Retain)) {
    llvm::Value *GetPropertyFn =
      CGM.getObjCRuntime().GetPropertyGetFunction();

    if (!GetPropertyFn) {
      CGM.ErrorUnsupported(PID, "Obj-C getter requiring atomic copy");
      FinishFunction();
      return;
    }

    // Return (ivar-type) objc_getProperty((id) self, _cmd, offset, true).
    // FIXME: Can't this be simpler? This might even be worse than the
    // corresponding gcc code.
    CodeGenTypes &Types = CGM.getTypes();
    ValueDecl *Cmd = OMD->getCmdDecl();
    llvm::Value *CmdVal = Builder.CreateLoad(LocalDeclMap[Cmd], "cmd");
    QualType IdTy = getContext().getObjCIdType();
    llvm::Value *SelfAsId =
      Builder.CreateBitCast(LoadObjCSelf(), Types.ConvertType(IdTy));
    llvm::Value *Offset = EmitIvarOffset(IMP->getClassInterface(), Ivar);
    llvm::Value *True =
      llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 1);
    CallArgList Args;
    Args.push_back(std::make_pair(RValue::get(SelfAsId), IdTy));
    Args.push_back(std::make_pair(RValue::get(CmdVal), Cmd->getType()));
    Args.push_back(std::make_pair(RValue::get(Offset),
                getContext().getPointerDiffType()));
    Args.push_back(std::make_pair(RValue::get(True), getContext().BoolTy));
    // FIXME: We shouldn't need to get the function info here, the
    // runtime already should have computed it to build the function.
    RValue RV = EmitCall(Types.getFunctionInfo(PD->getType(), Args,
                                               FunctionType::ExtInfo()),
                         GetPropertyFn, ReturnValueSlot(), Args);
    // We need to fix the type here. Ivars with copy & retain are
    // always objects so we don't need to worry about complex or
    // aggregates.
    RV = RValue::get(Builder.CreateBitCast(RV.getScalarVal(),
                                           Types.ConvertType(PD->getType())));
    EmitReturnOfRValue(RV, PD->getType());
  } else {
    const llvm::Triple &Triple = getContext().Target.getTriple();
    QualType IVART = Ivar->getType();
    if (IsAtomic &&
        IVART->isScalarType() &&
        (Triple.getArch() == llvm::Triple::arm ||
         Triple.getArch() == llvm::Triple::thumb) &&
        (getContext().getTypeSizeInChars(IVART) 
         > CharUnits::fromQuantity(4)) &&
        CGM.getObjCRuntime().GetGetStructFunction()) {
      GenerateObjCGetterBody(Ivar, true, false);
    }
    else if (IVART->isAnyComplexType()) {
      LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), 
                                    Ivar, 0);
      ComplexPairTy Pair = LoadComplexFromAddr(LV.getAddress(),
                                               LV.isVolatileQualified());
      StoreComplexToAddr(Pair, ReturnValue, LV.isVolatileQualified());
    }
    else if (hasAggregateLLVMType(IVART)) {
      bool IsStrong = false;
      if ((IsAtomic || (IsStrong = IvarTypeWithAggrGCObjects(IVART)))
          && CurFnInfo->getReturnInfo().getKind() == ABIArgInfo::Indirect
          && CGM.getObjCRuntime().GetGetStructFunction()) {
        GenerateObjCGetterBody(Ivar, IsAtomic, IsStrong);
      }
      else {
        if (PID->getGetterCXXConstructor()) {
          ReturnStmt *Stmt = 
            new (getContext()) ReturnStmt(SourceLocation(), 
                                          PID->getGetterCXXConstructor(),
                                          0);
          EmitReturnStmt(*Stmt);
        }
        else {
          LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), 
                                        Ivar, 0);
          EmitAggregateCopy(ReturnValue, LV.getAddress(), IVART);
        }
      }
    } 
    else {
        LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), 
                                    Ivar, 0);
        CodeGenTypes &Types = CGM.getTypes();
        RValue RV = EmitLoadOfLValue(LV, IVART);
        RV = RValue::get(Builder.CreateBitCast(RV.getScalarVal(),
                                               Types.ConvertType(PD->getType())));
        EmitReturnOfRValue(RV, PD->getType());
    }
  }

  FinishFunction();
}

void CodeGenFunction::GenerateObjCAtomicSetterBody(ObjCMethodDecl *OMD,
                                                   ObjCIvarDecl *Ivar) {
  // objc_copyStruct (&structIvar, &Arg, 
  //                  sizeof (struct something), true, false);
  llvm::Value *GetCopyStructFn =
  CGM.getObjCRuntime().GetSetStructFunction();
  CodeGenTypes &Types = CGM.getTypes();
  CallArgList Args;
  LValue LV = EmitLValueForIvar(TypeOfSelfObject(), LoadObjCSelf(), Ivar, 0);
  RValue RV =
    RValue::get(Builder.CreateBitCast(LV.getAddress(),
                Types.ConvertType(getContext().VoidPtrTy)));
  Args.push_back(std::make_pair(RV, getContext().VoidPtrTy));
  llvm::Value *Arg = LocalDeclMap[*OMD->param_begin()];
  llvm::Value *ArgAsPtrTy =
  Builder.CreateBitCast(Arg,
                      Types.ConvertType(getContext().VoidPtrTy));
  RV = RValue::get(ArgAsPtrTy);
  Args.push_back(std::make_pair(RV, getContext().VoidPtrTy));
  // sizeof (Type of Ivar)
  CharUnits Size =  getContext().getTypeSizeInChars(Ivar->getType());
  llvm::Value *SizeVal =
  llvm::ConstantInt::get(Types.ConvertType(getContext().LongTy), 
                         Size.getQuantity());
  Args.push_back(std::make_pair(RValue::get(SizeVal),
                                getContext().LongTy));
  llvm::Value *True =
  llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 1);
  Args.push_back(std::make_pair(RValue::get(True), getContext().BoolTy));
  llvm::Value *False =
  llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 0);
  Args.push_back(std::make_pair(RValue::get(False), getContext().BoolTy));
  EmitCall(Types.getFunctionInfo(getContext().VoidTy, Args,
                                 FunctionType::ExtInfo()),
           GetCopyStructFn, ReturnValueSlot(), Args);
}

/// GenerateObjCSetter - Generate an Objective-C property setter
/// function. The given Decl must be an ObjCImplementationDecl. @synthesize
/// is illegal within a category.
void CodeGenFunction::GenerateObjCSetter(ObjCImplementationDecl *IMP,
                                         const ObjCPropertyImplDecl *PID) {
  ObjCIvarDecl *Ivar = PID->getPropertyIvarDecl();
  const ObjCPropertyDecl *PD = PID->getPropertyDecl();
  ObjCMethodDecl *OMD = PD->getSetterMethodDecl();
  assert(OMD && "Invalid call to generate setter (empty method)");
  StartObjCMethod(OMD, IMP->getClassInterface());

  bool IsCopy = PD->getSetterKind() == ObjCPropertyDecl::Copy;
  bool IsAtomic =
    !(PD->getPropertyAttributes() & ObjCPropertyDecl::OBJC_PR_nonatomic);

  // Determine if we should use an objc_setProperty call for
  // this. Properties with 'copy' semantics always use it, as do
  // non-atomic properties with 'release' semantics as long as we are
  // not in gc-only mode.
  if (IsCopy ||
      (CGM.getLangOptions().getGCMode() != LangOptions::GCOnly &&
       PD->getSetterKind() == ObjCPropertyDecl::Retain)) {
    llvm::Value *SetPropertyFn =
      CGM.getObjCRuntime().GetPropertySetFunction();

    if (!SetPropertyFn) {
      CGM.ErrorUnsupported(PID, "Obj-C getter requiring atomic copy");
      FinishFunction();
      return;
    }

    // Emit objc_setProperty((id) self, _cmd, offset, arg,
    //                       <is-atomic>, <is-copy>).
    // FIXME: Can't this be simpler? This might even be worse than the
    // corresponding gcc code.
    CodeGenTypes &Types = CGM.getTypes();
    ValueDecl *Cmd = OMD->getCmdDecl();
    llvm::Value *CmdVal = Builder.CreateLoad(LocalDeclMap[Cmd], "cmd");
    QualType IdTy = getContext().getObjCIdType();
    llvm::Value *SelfAsId =
      Builder.CreateBitCast(LoadObjCSelf(), Types.ConvertType(IdTy));
    llvm::Value *Offset = EmitIvarOffset(IMP->getClassInterface(), Ivar);
    llvm::Value *Arg = LocalDeclMap[*OMD->param_begin()];
    llvm::Value *ArgAsId =
      Builder.CreateBitCast(Builder.CreateLoad(Arg, "arg"),
                            Types.ConvertType(IdTy));
    llvm::Value *True =
      llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 1);
    llvm::Value *False =
      llvm::ConstantInt::get(Types.ConvertType(getContext().BoolTy), 0);
    CallArgList Args;
    Args.push_back(std::make_pair(RValue::get(SelfAsId), IdTy));
    Args.push_back(std::make_pair(RValue::get(CmdVal), Cmd->getType()));
    Args.push_back(std::make_pair(RValue::get(Offset),
                getContext().getPointerDiffType()));
    Args.push_back(std::make_pair(RValue::get(ArgAsId), IdTy));
    Args.push_back(std::make_pair(RValue::get(IsAtomic ? True : False),
                                  getContext().BoolTy));
    Args.push_back(std::make_pair(RValue::get(IsCopy ? True : False),
                                  getContext().BoolTy));
    // FIXME: We shouldn't need to get the function info here, the runtime
    // already should have computed it to build the function.
    EmitCall(Types.getFunctionInfo(getContext().VoidTy, Args,
                                   FunctionType::ExtInfo()),
             SetPropertyFn,
             ReturnValueSlot(), Args);
  } else if (IsAtomic && hasAggregateLLVMType(Ivar->getType()) &&
             !Ivar->getType()->isAnyComplexType() &&
             IndirectObjCSetterArg(*CurFnInfo)
             && CGM.getObjCRuntime().GetSetStructFunction()) {
    // objc_copyStruct (&structIvar, &Arg, 
    //                  sizeof (struct something), true, false);
    GenerateObjCAtomicSetterBody(OMD, Ivar);
  } else if (PID->getSetterCXXAssignment()) {
    EmitIgnoredExpr(PID->getSetterCXXAssignment());
  } else {
    const llvm::Triple &Triple = getContext().Target.getTriple();
    QualType IVART = Ivar->getType();
    if (IsAtomic &&
        IVART->isScalarType() &&
        (Triple.getArch() == llvm::Triple::arm ||
         Triple.getArch() == llvm::Triple::thumb) &&
        (getContext().getTypeSizeInChars(IVART)
          > CharUnits::fromQuantity(4)) &&
        CGM.getObjCRuntime().GetGetStructFunction()) {
      GenerateObjCAtomicSetterBody(OMD, Ivar);
    }
    else {
      // FIXME: Find a clean way to avoid AST node creation.
      SourceLocation Loc = PD->getLocation();
      ValueDecl *Self = OMD->getSelfDecl();
      ObjCIvarDecl *Ivar = PID->getPropertyIvarDecl();
      DeclRefExpr Base(Self, Self->getType(), VK_RValue, Loc);
      ParmVarDecl *ArgDecl = *OMD->param_begin();
      DeclRefExpr Arg(ArgDecl, ArgDecl->getType(), VK_LValue, Loc);
      ObjCIvarRefExpr IvarRef(Ivar, Ivar->getType(), Loc, &Base, true, true);
    
      // The property type can differ from the ivar type in some situations with
      // Objective-C pointer types, we can always bit cast the RHS in these cases.
      if (getContext().getCanonicalType(Ivar->getType()) !=
          getContext().getCanonicalType(ArgDecl->getType())) {
        ImplicitCastExpr ArgCasted(ImplicitCastExpr::OnStack,
                                   Ivar->getType(), CK_BitCast, &Arg,
                                   VK_RValue);
        BinaryOperator Assign(&IvarRef, &ArgCasted, BO_Assign,
                              Ivar->getType(), VK_RValue, OK_Ordinary, Loc);
        EmitStmt(&Assign);
      } else {
        BinaryOperator Assign(&IvarRef, &Arg, BO_Assign,
                              Ivar->getType(), VK_RValue, OK_Ordinary, Loc);
        EmitStmt(&Assign);
      }
    }
  }

  FinishFunction();
}

void CodeGenFunction::GenerateObjCCtorDtorMethod(ObjCImplementationDecl *IMP,
                                                 ObjCMethodDecl *MD,
                                                 bool ctor) {
  llvm::SmallVector<CXXCtorInitializer *, 8> IvarInitializers;
  MD->createImplicitParams(CGM.getContext(), IMP->getClassInterface());
  StartObjCMethod(MD, IMP->getClassInterface());
  for (ObjCImplementationDecl::init_const_iterator B = IMP->init_begin(),
       E = IMP->init_end(); B != E; ++B) {
    CXXCtorInitializer *Member = (*B);
    IvarInitializers.push_back(Member);
  }
  if (ctor) {
    for (unsigned I = 0, E = IvarInitializers.size(); I != E; ++I) {
      CXXCtorInitializer *IvarInit = IvarInitializers[I];
      FieldDecl *Field = IvarInit->getAnyMember();
      ObjCIvarDecl  *Ivar = cast<ObjCIvarDecl>(Field);
      LValue LV = EmitLValueForIvar(TypeOfSelfObject(), 
                                    LoadObjCSelf(), Ivar, 0);
      EmitAggExpr(IvarInit->getInit(), AggValueSlot::forLValue(LV, true));
    }
    // constructor returns 'self'.
    CodeGenTypes &Types = CGM.getTypes();
    QualType IdTy(CGM.getContext().getObjCIdType());
    llvm::Value *SelfAsId =
      Builder.CreateBitCast(LoadObjCSelf(), Types.ConvertType(IdTy));
    EmitReturnOfRValue(RValue::get(SelfAsId), IdTy);
  } else {
    // dtor
    for (size_t i = IvarInitializers.size(); i > 0; --i) {
      FieldDecl *Field = IvarInitializers[i - 1]->getAnyMember();
      QualType FieldType = Field->getType();
      const ConstantArrayType *Array = 
        getContext().getAsConstantArrayType(FieldType);
      if (Array)
        FieldType = getContext().getBaseElementType(FieldType);
      
      ObjCIvarDecl  *Ivar = cast<ObjCIvarDecl>(Field);
      LValue LV = EmitLValueForIvar(TypeOfSelfObject(), 
                                    LoadObjCSelf(), Ivar, 0);
      const RecordType *RT = FieldType->getAs<RecordType>();
      CXXRecordDecl *FieldClassDecl = cast<CXXRecordDecl>(RT->getDecl());
      CXXDestructorDecl *Dtor = FieldClassDecl->getDestructor();
      if (!Dtor->isTrivial()) {
        if (Array) {
          const llvm::Type *BasePtr = ConvertType(FieldType);
          BasePtr = llvm::PointerType::getUnqual(BasePtr);
          llvm::Value *BaseAddrPtr =
            Builder.CreateBitCast(LV.getAddress(), BasePtr);
          EmitCXXAggrDestructorCall(Dtor,
                                    Array, BaseAddrPtr);
        } else {
          EmitCXXDestructorCall(Dtor,
                                Dtor_Complete, /*ForVirtualBase=*/false,
                                LV.getAddress());
        }
      }
    }
  }
  FinishFunction();
}

bool CodeGenFunction::IndirectObjCSetterArg(const CGFunctionInfo &FI) {
  CGFunctionInfo::const_arg_iterator it = FI.arg_begin();
  it++; it++;
  const ABIArgInfo &AI = it->info;
  // FIXME. Is this sufficient check?
  return (AI.getKind() == ABIArgInfo::Indirect);
}

bool CodeGenFunction::IvarTypeWithAggrGCObjects(QualType Ty) {
  if (CGM.getLangOptions().getGCMode() == LangOptions::NonGC)
    return false;
  if (const RecordType *FDTTy = Ty.getTypePtr()->getAs<RecordType>())
    return FDTTy->getDecl()->hasObjectMember();
  return false;
}

llvm::Value *CodeGenFunction::LoadObjCSelf() {
  const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
  return Builder.CreateLoad(LocalDeclMap[OMD->getSelfDecl()], "self");
}

QualType CodeGenFunction::TypeOfSelfObject() {
  const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CurFuncDecl);
  ImplicitParamDecl *selfDecl = OMD->getSelfDecl();
  const ObjCObjectPointerType *PTy = cast<ObjCObjectPointerType>(
    getContext().getCanonicalType(selfDecl->getType()));
  return PTy->getPointeeType();
}

LValue
CodeGenFunction::EmitObjCPropertyRefLValue(const ObjCPropertyRefExpr *E) {
  // This is a special l-value that just issues sends when we load or
  // store through it.

  // For certain base kinds, we need to emit the base immediately.
  llvm::Value *Base;
  if (E->isSuperReceiver())
    Base = LoadObjCSelf();
  else if (E->isClassReceiver())
    Base = CGM.getObjCRuntime().GetClass(Builder, E->getClassReceiver());
  else
    Base = EmitScalarExpr(E->getBase());
  return LValue::MakePropertyRef(E, Base);
}

static RValue GenerateMessageSendSuper(CodeGenFunction &CGF,
                                       ReturnValueSlot Return,
                                       QualType ResultType,
                                       Selector S,
                                       llvm::Value *Receiver,
                                       const CallArgList &CallArgs) {
  const ObjCMethodDecl *OMD = cast<ObjCMethodDecl>(CGF.CurFuncDecl);
  bool isClassMessage = OMD->isClassMethod();
  bool isCategoryImpl = isa<ObjCCategoryImplDecl>(OMD->getDeclContext());
  return CGF.CGM.getObjCRuntime()
                .GenerateMessageSendSuper(CGF, Return, ResultType,
                                          S, OMD->getClassInterface(),
                                          isCategoryImpl, Receiver,
                                          isClassMessage, CallArgs);
}

RValue CodeGenFunction::EmitLoadOfPropertyRefLValue(LValue LV,
                                                    ReturnValueSlot Return) {
  const ObjCPropertyRefExpr *E = LV.getPropertyRefExpr();
  QualType ResultType;
  Selector S;
  if (E->isExplicitProperty()) {
    const ObjCPropertyDecl *Property = E->getExplicitProperty();
    S = Property->getGetterName();
    ResultType = E->getType();
  } else {
    const ObjCMethodDecl *Getter = E->getImplicitPropertyGetter();
    S = Getter->getSelector();
    ResultType = Getter->getResultType(); // with reference!
  }

  llvm::Value *Receiver = LV.getPropertyRefBaseAddr();

  // Accesses to 'super' follow a different code path.
  if (E->isSuperReceiver())
    return GenerateMessageSendSuper(*this, Return, ResultType,
                                    S, Receiver, CallArgList());

  const ObjCInterfaceDecl *ReceiverClass
    = (E->isClassReceiver() ? E->getClassReceiver() : 0);
  return CGM.getObjCRuntime().
             GenerateMessageSend(*this, Return, ResultType, S,
                                 Receiver, CallArgList(), ReceiverClass);
}

void CodeGenFunction::EmitStoreThroughPropertyRefLValue(RValue Src,
                                                        LValue Dst) {
  const ObjCPropertyRefExpr *E = Dst.getPropertyRefExpr();
  Selector S = E->getSetterSelector();
  QualType ArgType;
  if (E->isImplicitProperty()) {
    const ObjCMethodDecl *Setter = E->getImplicitPropertySetter();
    ObjCMethodDecl::param_iterator P = Setter->param_begin(); 
    ArgType = (*P)->getType();
  } else {
    ArgType = E->getType();
  }
  // FIXME. Other than scalars, AST is not adequate for setter and
  // getter type mismatches which require conversion.
  if (Src.isScalar()) {
    llvm::Value *SrcVal = Src.getScalarVal();
    QualType DstType = getContext().getCanonicalType(ArgType);
    const llvm::Type *DstTy = ConvertType(DstType);
    if (SrcVal->getType() != DstTy)
      Src = 
        RValue::get(EmitScalarConversion(SrcVal, E->getType(), DstType));
  }
  
  CallArgList Args;
  Args.push_back(std::make_pair(Src, ArgType));

  llvm::Value *Receiver = Dst.getPropertyRefBaseAddr();
  QualType ResultType = getContext().VoidTy;

  if (E->isSuperReceiver()) {
    GenerateMessageSendSuper(*this, ReturnValueSlot(),
                             ResultType, S, Receiver, Args);
    return;
  }

  const ObjCInterfaceDecl *ReceiverClass
    = (E->isClassReceiver() ? E->getClassReceiver() : 0);

  CGM.getObjCRuntime().GenerateMessageSend(*this, ReturnValueSlot(),
                                           ResultType, S, Receiver, Args,
                                           ReceiverClass);
}

void CodeGenFunction::EmitObjCForCollectionStmt(const ObjCForCollectionStmt &S){
  llvm::Constant *EnumerationMutationFn =
    CGM.getObjCRuntime().EnumerationMutationFunction();

  if (!EnumerationMutationFn) {
    CGM.ErrorUnsupported(&S, "Obj-C fast enumeration for this runtime");
    return;
  }

  // The local variable comes into scope immediately.
  AutoVarEmission variable = AutoVarEmission::invalid();
  if (const DeclStmt *SD = dyn_cast<DeclStmt>(S.getElement()))
    variable = EmitAutoVarAlloca(*cast<VarDecl>(SD->getSingleDecl()));

  CGDebugInfo *DI = getDebugInfo();
  if (DI) {
    DI->setLocation(S.getSourceRange().getBegin());
    DI->EmitRegionStart(Builder);
  }

  JumpDest LoopEnd = getJumpDestInCurrentScope("forcoll.end");
  JumpDest AfterBody = getJumpDestInCurrentScope("forcoll.next");

  // Fast enumeration state.
  QualType StateTy = getContext().getObjCFastEnumerationStateType();
  llvm::Value *StatePtr = CreateMemTemp(StateTy, "state.ptr");
  EmitNullInitialization(StatePtr, StateTy);

  // Number of elements in the items array.
  static const unsigned NumItems = 16;

  // Fetch the countByEnumeratingWithState:objects:count: selector.
  IdentifierInfo *II[] = {
    &CGM.getContext().Idents.get("countByEnumeratingWithState"),
    &CGM.getContext().Idents.get("objects"),
    &CGM.getContext().Idents.get("count")
  };
  Selector FastEnumSel =
    CGM.getContext().Selectors.getSelector(llvm::array_lengthof(II), &II[0]);

  QualType ItemsTy =
    getContext().getConstantArrayType(getContext().getObjCIdType(),
                                      llvm::APInt(32, NumItems),
                                      ArrayType::Normal, 0);
  llvm::Value *ItemsPtr = CreateMemTemp(ItemsTy, "items.ptr");

  // Emit the collection pointer.
  llvm::Value *Collection = EmitScalarExpr(S.getCollection());

  // Send it our message:
  CallArgList Args;

  // The first argument is a temporary of the enumeration-state type.
  Args.push_back(std::make_pair(RValue::get(StatePtr),
                                getContext().getPointerType(StateTy)));

  // The second argument is a temporary array with space for NumItems
  // pointers.  We'll actually be loading elements from the array
  // pointer written into the control state; this buffer is so that
  // collections that *aren't* backed by arrays can still queue up
  // batches of elements.
  Args.push_back(std::make_pair(RValue::get(ItemsPtr),
                                getContext().getPointerType(ItemsTy)));

  // The third argument is the capacity of that temporary array.
  const llvm::Type *UnsignedLongLTy = ConvertType(getContext().UnsignedLongTy);
  llvm::Constant *Count = llvm::ConstantInt::get(UnsignedLongLTy, NumItems);
  Args.push_back(std::make_pair(RValue::get(Count),
                                getContext().UnsignedLongTy));

  // Start the enumeration.
  RValue CountRV =
    CGM.getObjCRuntime().GenerateMessageSend(*this, ReturnValueSlot(),
                                             getContext().UnsignedLongTy,
                                             FastEnumSel,
                                             Collection, Args);

  // The initial number of objects that were returned in the buffer.
  llvm::Value *initialBufferLimit = CountRV.getScalarVal();

  llvm::BasicBlock *EmptyBB = createBasicBlock("forcoll.empty");
  llvm::BasicBlock *LoopInitBB = createBasicBlock("forcoll.loopinit");

  llvm::Value *zero = llvm::Constant::getNullValue(UnsignedLongLTy);

  // If the limit pointer was zero to begin with, the collection is
  // empty; skip all this.
  Builder.CreateCondBr(Builder.CreateICmpEQ(initialBufferLimit, zero, "iszero"),
                       EmptyBB, LoopInitBB);

  // Otherwise, initialize the loop.
  EmitBlock(LoopInitBB);

  // Save the initial mutations value.  This is the value at an
  // address that was written into the state object by
  // countByEnumeratingWithState:objects:count:.
  llvm::Value *StateMutationsPtrPtr =
    Builder.CreateStructGEP(StatePtr, 2, "mutationsptr.ptr");
  llvm::Value *StateMutationsPtr = Builder.CreateLoad(StateMutationsPtrPtr,
                                                      "mutationsptr");

  llvm::Value *initialMutations =
    Builder.CreateLoad(StateMutationsPtr, "forcoll.initial-mutations");

  // Start looping.  This is the point we return to whenever we have a
  // fresh, non-empty batch of objects.
  llvm::BasicBlock *LoopBodyBB = createBasicBlock("forcoll.loopbody");
  EmitBlock(LoopBodyBB);

  // The current index into the buffer.
  llvm::PHINode *index = Builder.CreatePHI(UnsignedLongLTy, "forcoll.index");
  index->addIncoming(zero, LoopInitBB);

  // The current buffer size.
  llvm::PHINode *count = Builder.CreatePHI(UnsignedLongLTy, "forcoll.count");
  count->addIncoming(initialBufferLimit, LoopInitBB);

  // Check whether the mutations value has changed from where it was
  // at start.  StateMutationsPtr should actually be invariant between
  // refreshes.
  StateMutationsPtr = Builder.CreateLoad(StateMutationsPtrPtr, "mutationsptr");
  llvm::Value *currentMutations
    = Builder.CreateLoad(StateMutationsPtr, "statemutations");

  llvm::BasicBlock *WasMutatedBB = createBasicBlock("forcoll.mutated");
  llvm::BasicBlock *WasNotMutatedBB = createBasicBlock("forcoll.notmutated");

  Builder.CreateCondBr(Builder.CreateICmpEQ(currentMutations, initialMutations),
                       WasNotMutatedBB, WasMutatedBB);

  // If so, call the enumeration-mutation function.
  EmitBlock(WasMutatedBB);
  llvm::Value *V =
    Builder.CreateBitCast(Collection,
                          ConvertType(getContext().getObjCIdType()),
                          "tmp");
  CallArgList Args2;
  Args2.push_back(std::make_pair(RValue::get(V),
                                getContext().getObjCIdType()));
  // FIXME: We shouldn't need to get the function info here, the runtime already
  // should have computed it to build the function.
  EmitCall(CGM.getTypes().getFunctionInfo(getContext().VoidTy, Args2,
                                          FunctionType::ExtInfo()),
           EnumerationMutationFn, ReturnValueSlot(), Args2);

  // Otherwise, or if the mutation function returns, just continue.
  EmitBlock(WasNotMutatedBB);

  // Initialize the element variable.
  RunCleanupsScope elementVariableScope(*this);
  bool elementIsVariable;
  LValue elementLValue;
  QualType elementType;
  if (const DeclStmt *SD = dyn_cast<DeclStmt>(S.getElement())) {
    // Initialize the variable, in case it's a __block variable or something.
    EmitAutoVarInit(variable);

    const VarDecl* D = cast<VarDecl>(SD->getSingleDecl());
    DeclRefExpr tempDRE(const_cast<VarDecl*>(D), D->getType(),
                        VK_LValue, SourceLocation());
    elementLValue = EmitLValue(&tempDRE);
    elementType = D->getType();
    elementIsVariable = true;
  } else {
    elementLValue = LValue(); // suppress warning
    elementType = cast<Expr>(S.getElement())->getType();
    elementIsVariable = false;
  }
  const llvm::Type *convertedElementType = ConvertType(elementType);

  // Fetch the buffer out of the enumeration state.
  // TODO: this pointer should actually be invariant between
  // refreshes, which would help us do certain loop optimizations.
  llvm::Value *StateItemsPtr =
    Builder.CreateStructGEP(StatePtr, 1, "stateitems.ptr");
  llvm::Value *EnumStateItems =
    Builder.CreateLoad(StateItemsPtr, "stateitems");

  // Fetch the value at the current index from the buffer.
  llvm::Value *CurrentItemPtr =
    Builder.CreateGEP(EnumStateItems, index, "currentitem.ptr");
  llvm::Value *CurrentItem = Builder.CreateLoad(CurrentItemPtr);

  // Cast that value to the right type.
  CurrentItem = Builder.CreateBitCast(CurrentItem, convertedElementType,
                                      "currentitem");

  // Make sure we have an l-value.  Yes, this gets evaluated every
  // time through the loop.
  if (!elementIsVariable)
    elementLValue = EmitLValue(cast<Expr>(S.getElement()));

  EmitStoreThroughLValue(RValue::get(CurrentItem), elementLValue, elementType);

  // If we do have an element variable, this assignment is the end of
  // its initialization.
  if (elementIsVariable)
    EmitAutoVarCleanups(variable);

  // Perform the loop body, setting up break and continue labels.
  BreakContinueStack.push_back(BreakContinue(LoopEnd, AfterBody));
  {
    RunCleanupsScope Scope(*this);
    EmitStmt(S.getBody());
  }
  BreakContinueStack.pop_back();

  // Destroy the element variable now.
  elementVariableScope.ForceCleanup();

  // Check whether there are more elements.
  EmitBlock(AfterBody.getBlock());

  llvm::BasicBlock *FetchMoreBB = createBasicBlock("forcoll.refetch");

  // First we check in the local buffer.
  llvm::Value *indexPlusOne
    = Builder.CreateAdd(index, llvm::ConstantInt::get(UnsignedLongLTy, 1));

  // If we haven't overrun the buffer yet, we can continue.
  Builder.CreateCondBr(Builder.CreateICmpULT(indexPlusOne, count),
                       LoopBodyBB, FetchMoreBB);

  index->addIncoming(indexPlusOne, AfterBody.getBlock());
  count->addIncoming(count, AfterBody.getBlock());

  // Otherwise, we have to fetch more elements.
  EmitBlock(FetchMoreBB);

  CountRV =
    CGM.getObjCRuntime().GenerateMessageSend(*this, ReturnValueSlot(),
                                             getContext().UnsignedLongTy,
                                             FastEnumSel,
                                             Collection, Args);

  // If we got a zero count, we're done.
  llvm::Value *refetchCount = CountRV.getScalarVal();

  // (note that the message send might split FetchMoreBB)
  index->addIncoming(zero, Builder.GetInsertBlock());
  count->addIncoming(refetchCount, Builder.GetInsertBlock());

  Builder.CreateCondBr(Builder.CreateICmpEQ(refetchCount, zero),
                       EmptyBB, LoopBodyBB);

  // No more elements.
  EmitBlock(EmptyBB);

  if (!elementIsVariable) {
    // If the element was not a declaration, set it to be null.

    llvm::Value *null = llvm::Constant::getNullValue(convertedElementType);
    elementLValue = EmitLValue(cast<Expr>(S.getElement()));
    EmitStoreThroughLValue(RValue::get(null), elementLValue, elementType);
  }

  if (DI) {
    DI->setLocation(S.getSourceRange().getEnd());
    DI->EmitRegionEnd(Builder);
  }

  EmitBlock(LoopEnd.getBlock());
}

void CodeGenFunction::EmitObjCAtTryStmt(const ObjCAtTryStmt &S) {
  CGM.getObjCRuntime().EmitTryStmt(*this, S);
}

void CodeGenFunction::EmitObjCAtThrowStmt(const ObjCAtThrowStmt &S) {
  CGM.getObjCRuntime().EmitThrowStmt(*this, S);
}

void CodeGenFunction::EmitObjCAtSynchronizedStmt(
                                              const ObjCAtSynchronizedStmt &S) {
  CGM.getObjCRuntime().EmitSynchronizedStmt(*this, S);
}

CGObjCRuntime::~CGObjCRuntime() {}
