//===----- CGCUDANV.cpp - Interface to NVIDIA CUDA Runtime ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for CUDA code generation targeting the NVIDIA CUDA
// runtime library.
//
//===----------------------------------------------------------------------===//

#include "CGCUDARuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/CodeGen/ConstantInitBuilder.h"
#include "clang/AST/Decl.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"

using namespace clang;
using namespace CodeGen;

namespace {

class CGNVCUDARuntime : public CGCUDARuntime {

private:
  llvm::IntegerType *IntTy, *SizeTy;
  llvm::Type *VoidTy;
  llvm::PointerType *CharPtrTy, *VoidPtrTy, *VoidPtrPtrTy;

  /// Convenience reference to LLVM Context
  llvm::LLVMContext &Context;
  /// Convenience reference to the current module
  llvm::Module &TheModule;
  /// Keeps track of kernel launch stubs emitted in this module
  llvm::SmallVector<llvm::Function *, 16> EmittedKernels;
  llvm::SmallVector<std::pair<llvm::GlobalVariable *, unsigned>, 16> DeviceVars;
  /// Keeps track of variables containing handles of GPU binaries. Populated by
  /// ModuleCtorFunction() and used to create corresponding cleanup calls in
  /// ModuleDtorFunction()
  llvm::SmallVector<llvm::GlobalVariable *, 16> GpuBinaryHandles;

  llvm::Constant *getSetupArgumentFn() const;
  llvm::Constant *getLaunchFn() const;

  /// Creates a function to register all kernel stubs generated in this module.
  llvm::Function *makeRegisterGlobalsFn();

  /// Helper function that generates a constant string and returns a pointer to
  /// the start of the string.  The result of this function can be used anywhere
  /// where the C code specifies const char*.
  llvm::Constant *makeConstantString(const std::string &Str,
                                     const std::string &Name = "",
                                     const std::string &SectionName = "",
                                     unsigned Alignment = 0) {
    llvm::Constant *Zeros[] = {llvm::ConstantInt::get(SizeTy, 0),
                               llvm::ConstantInt::get(SizeTy, 0)};
    auto ConstStr = CGM.GetAddrOfConstantCString(Str, Name.c_str());
    llvm::GlobalVariable *GV =
        cast<llvm::GlobalVariable>(ConstStr.getPointer());
    if (!SectionName.empty())
      GV->setSection(SectionName);
    if (Alignment)
      GV->setAlignment(Alignment);

    return llvm::ConstantExpr::getGetElementPtr(ConstStr.getElementType(),
                                                ConstStr.getPointer(), Zeros);
 }

  void emitDeviceStubBody(CodeGenFunction &CGF, FunctionArgList &Args);

public:
  CGNVCUDARuntime(CodeGenModule &CGM);

  void emitDeviceStub(CodeGenFunction &CGF, FunctionArgList &Args) override;
  void registerDeviceVar(llvm::GlobalVariable &Var, unsigned Flags) override {
    DeviceVars.push_back(std::make_pair(&Var, Flags));
  }

  /// Creates module constructor function
  llvm::Function *makeModuleCtorFunction() override;
  /// Creates module destructor function
  llvm::Function *makeModuleDtorFunction() override;
};

}

CGNVCUDARuntime::CGNVCUDARuntime(CodeGenModule &CGM)
    : CGCUDARuntime(CGM), Context(CGM.getLLVMContext()),
      TheModule(CGM.getModule()) {
  CodeGen::CodeGenTypes &Types = CGM.getTypes();
  ASTContext &Ctx = CGM.getContext();

  IntTy = CGM.IntTy;
  SizeTy = CGM.SizeTy;
  VoidTy = CGM.VoidTy;

  CharPtrTy = llvm::PointerType::getUnqual(Types.ConvertType(Ctx.CharTy));
  VoidPtrTy = cast<llvm::PointerType>(Types.ConvertType(Ctx.VoidPtrTy));
  VoidPtrPtrTy = VoidPtrTy->getPointerTo();
}

llvm::Constant *CGNVCUDARuntime::getSetupArgumentFn() const {
  // cudaError_t cudaSetupArgument(void *, size_t, size_t)
  llvm::Type *Params[] = {VoidPtrTy, SizeTy, SizeTy};
  return CGM.CreateRuntimeFunction(llvm::FunctionType::get(IntTy,
                                                           Params, false),
                                   "cudaSetupArgument");
}

llvm::Constant *CGNVCUDARuntime::getLaunchFn() const {
  // cudaError_t cudaLaunch(char *)
  return CGM.CreateRuntimeFunction(
      llvm::FunctionType::get(IntTy, CharPtrTy, false), "cudaLaunch");
}

void CGNVCUDARuntime::emitDeviceStub(CodeGenFunction &CGF,
                                     FunctionArgList &Args) {
  EmittedKernels.push_back(CGF.CurFn);
  emitDeviceStubBody(CGF, Args);
}

void CGNVCUDARuntime::emitDeviceStubBody(CodeGenFunction &CGF,
                                         FunctionArgList &Args) {
  // Emit a call to cudaSetupArgument for each arg in Args.
  llvm::Constant *cudaSetupArgFn = getSetupArgumentFn();
  llvm::BasicBlock *EndBlock = CGF.createBasicBlock("setup.end");
  CharUnits Offset = CharUnits::Zero();
  for (const VarDecl *A : Args) {
    CharUnits TyWidth, TyAlign;
    std::tie(TyWidth, TyAlign) =
        CGM.getContext().getTypeInfoInChars(A->getType());
    Offset = Offset.alignTo(TyAlign);
    llvm::Value *Args[] = {
        CGF.Builder.CreatePointerCast(CGF.GetAddrOfLocalVar(A).getPointer(),
                                      VoidPtrTy),
        llvm::ConstantInt::get(SizeTy, TyWidth.getQuantity()),
        llvm::ConstantInt::get(SizeTy, Offset.getQuantity()),
    };
    llvm::CallSite CS = CGF.EmitRuntimeCallOrInvoke(cudaSetupArgFn, Args);
    llvm::Constant *Zero = llvm::ConstantInt::get(IntTy, 0);
    llvm::Value *CSZero = CGF.Builder.CreateICmpEQ(CS.getInstruction(), Zero);
    llvm::BasicBlock *NextBlock = CGF.createBasicBlock("setup.next");
    CGF.Builder.CreateCondBr(CSZero, NextBlock, EndBlock);
    CGF.EmitBlock(NextBlock);
    Offset += TyWidth;
  }

  // Emit the call to cudaLaunch
  llvm::Constant *cudaLaunchFn = getLaunchFn();
  llvm::Value *Arg = CGF.Builder.CreatePointerCast(CGF.CurFn, CharPtrTy);
  CGF.EmitRuntimeCallOrInvoke(cudaLaunchFn, Arg);
  CGF.EmitBranch(EndBlock);

  CGF.EmitBlock(EndBlock);
}

/// Creates a function that sets up state on the host side for CUDA objects that
/// have a presence on both the host and device sides. Specifically, registers
/// the host side of kernel functions and device global variables with the CUDA
/// runtime.
/// \code
/// void __cuda_register_globals(void** GpuBinaryHandle) {
///    __cudaRegisterFunction(GpuBinaryHandle,Kernel0,...);
///    ...
///    __cudaRegisterFunction(GpuBinaryHandle,KernelM,...);
///    __cudaRegisterVar(GpuBinaryHandle, GlobalVar0, ...);
///    ...
///    __cudaRegisterVar(GpuBinaryHandle, GlobalVarN, ...);
/// }
/// \endcode
llvm::Function *CGNVCUDARuntime::makeRegisterGlobalsFn() {
  // No need to register anything
  if (EmittedKernels.empty() && DeviceVars.empty())
    return nullptr;

  llvm::Function *RegisterKernelsFunc = llvm::Function::Create(
      llvm::FunctionType::get(VoidTy, VoidPtrPtrTy, false),
      llvm::GlobalValue::InternalLinkage, "__cuda_register_globals", &TheModule);
  llvm::BasicBlock *EntryBB =
      llvm::BasicBlock::Create(Context, "entry", RegisterKernelsFunc);
  CGBuilderTy Builder(CGM, Context);
  Builder.SetInsertPoint(EntryBB);

  // void __cudaRegisterFunction(void **, const char *, char *, const char *,
  //                             int, uint3*, uint3*, dim3*, dim3*, int*)
  llvm::Type *RegisterFuncParams[] = {
      VoidPtrPtrTy, CharPtrTy, CharPtrTy, CharPtrTy, IntTy,
      VoidPtrTy,    VoidPtrTy, VoidPtrTy, VoidPtrTy, IntTy->getPointerTo()};
  llvm::Constant *RegisterFunc = CGM.CreateRuntimeFunction(
      llvm::FunctionType::get(IntTy, RegisterFuncParams, false),
      "__cudaRegisterFunction");

  // Extract GpuBinaryHandle passed as the first argument passed to
  // __cuda_register_globals() and generate __cudaRegisterFunction() call for
  // each emitted kernel.
  llvm::Argument &GpuBinaryHandlePtr = *RegisterKernelsFunc->arg_begin();
  for (llvm::Function *Kernel : EmittedKernels) {
    llvm::Constant *KernelName = makeConstantString(Kernel->getName());
    llvm::Constant *NullPtr = llvm::ConstantPointerNull::get(VoidPtrTy);
    llvm::Value *Args[] = {
        &GpuBinaryHandlePtr, Builder.CreateBitCast(Kernel, VoidPtrTy),
        KernelName, KernelName, llvm::ConstantInt::get(IntTy, -1), NullPtr,
        NullPtr, NullPtr, NullPtr,
        llvm::ConstantPointerNull::get(IntTy->getPointerTo())};
    Builder.CreateCall(RegisterFunc, Args);
  }

  // void __cudaRegisterVar(void **, char *, char *, const char *,
  //                        int, int, int, int)
  llvm::Type *RegisterVarParams[] = {VoidPtrPtrTy, CharPtrTy, CharPtrTy,
                                     CharPtrTy,    IntTy,     IntTy,
                                     IntTy,        IntTy};
  llvm::Constant *RegisterVar = CGM.CreateRuntimeFunction(
      llvm::FunctionType::get(IntTy, RegisterVarParams, false),
      "__cudaRegisterVar");
  for (auto &Pair : DeviceVars) {
    llvm::GlobalVariable *Var = Pair.first;
    unsigned Flags = Pair.second;
    llvm::Constant *VarName = makeConstantString(Var->getName());
    uint64_t VarSize =
        CGM.getDataLayout().getTypeAllocSize(Var->getValueType());
    llvm::Value *Args[] = {
        &GpuBinaryHandlePtr,
        Builder.CreateBitCast(Var, VoidPtrTy),
        VarName,
        VarName,
        llvm::ConstantInt::get(IntTy, (Flags & ExternDeviceVar) ? 1 : 0),
        llvm::ConstantInt::get(IntTy, VarSize),
        llvm::ConstantInt::get(IntTy, (Flags & ConstantDeviceVar) ? 1 : 0),
        llvm::ConstantInt::get(IntTy, 0)};
    Builder.CreateCall(RegisterVar, Args);
  }

  Builder.CreateRetVoid();
  return RegisterKernelsFunc;
}

/// Creates a global constructor function for the module:
/// \code
/// void __cuda_module_ctor(void*) {
///     Handle0 = __cudaRegisterFatBinary(GpuBinaryBlob0);
///     __cuda_register_globals(Handle0);
///     ...
///     HandleN = __cudaRegisterFatBinary(GpuBinaryBlobN);
///     __cuda_register_globals(HandleN);
/// }
/// \endcode
llvm::Function *CGNVCUDARuntime::makeModuleCtorFunction() {
  // No need to generate ctors/dtors if there are no GPU binaries.
  if (CGM.getCodeGenOpts().CudaGpuBinaryFileNames.empty())
    return nullptr;

  // void __cuda_register_globals(void* handle);
  llvm::Function *RegisterGlobalsFunc = makeRegisterGlobalsFn();
  // void ** __cudaRegisterFatBinary(void *);
  llvm::Constant *RegisterFatbinFunc = CGM.CreateRuntimeFunction(
      llvm::FunctionType::get(VoidPtrPtrTy, VoidPtrTy, false),
      "__cudaRegisterFatBinary");
  // struct { int magic, int version, void * gpu_binary, void * dont_care };
  llvm::StructType *FatbinWrapperTy =
      llvm::StructType::get(IntTy, IntTy, VoidPtrTy, VoidPtrTy, nullptr);

  llvm::Function *ModuleCtorFunc = llvm::Function::Create(
      llvm::FunctionType::get(VoidTy, VoidPtrTy, false),
      llvm::GlobalValue::InternalLinkage, "__cuda_module_ctor", &TheModule);
  llvm::BasicBlock *CtorEntryBB =
      llvm::BasicBlock::Create(Context, "entry", ModuleCtorFunc);
  CGBuilderTy CtorBuilder(CGM, Context);

  CtorBuilder.SetInsertPoint(CtorEntryBB);

  // For each GPU binary, register it with the CUDA runtime and store returned
  // handle in a global variable and save the handle in GpuBinaryHandles vector
  // to be cleaned up in destructor on exit. Then associate all known kernels
  // with the GPU binary handle so CUDA runtime can figure out what to call on
  // the GPU side.
  for (const std::string &GpuBinaryFileName :
       CGM.getCodeGenOpts().CudaGpuBinaryFileNames) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> GpuBinaryOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(GpuBinaryFileName);
    if (std::error_code EC = GpuBinaryOrErr.getError()) {
      CGM.getDiags().Report(diag::err_cannot_open_file) << GpuBinaryFileName
                                                        << EC.message();
      continue;
    }

    const char *FatbinConstantName =
        CGM.getTriple().isMacOSX() ? "__NV_CUDA,__nv_fatbin" : ".nv_fatbin";
    // NVIDIA's cuobjdump looks for fatbins in this section.
    const char *FatbinSectionName =
        CGM.getTriple().isMacOSX() ? "__NV_CUDA,__fatbin" : ".nvFatBinSegment";

    // Create initialized wrapper structure that points to the loaded GPU binary
    ConstantInitBuilder Builder(CGM);
    auto Values = Builder.beginStruct(FatbinWrapperTy);
    // Fatbin wrapper magic.
    Values.addInt(IntTy, 0x466243b1);
    // Fatbin version.
    Values.addInt(IntTy, 1);
    // Data.
    Values.add(makeConstantString(GpuBinaryOrErr.get()->getBuffer(), 
                                  "", FatbinConstantName, 8));
    // Unused in fatbin v1.
    Values.add(llvm::ConstantPointerNull::get(VoidPtrTy));
    llvm::GlobalVariable *FatbinWrapper =
      Values.finishAndCreateGlobal("__cuda_fatbin_wrapper",
                                   CGM.getPointerAlign(),
                                   /*constant*/ true);
    FatbinWrapper->setSection(FatbinSectionName);

    // GpuBinaryHandle = __cudaRegisterFatBinary(&FatbinWrapper);
    llvm::CallInst *RegisterFatbinCall = CtorBuilder.CreateCall(
        RegisterFatbinFunc,
        CtorBuilder.CreateBitCast(FatbinWrapper, VoidPtrTy));
    llvm::GlobalVariable *GpuBinaryHandle = new llvm::GlobalVariable(
        TheModule, VoidPtrPtrTy, false, llvm::GlobalValue::InternalLinkage,
        llvm::ConstantPointerNull::get(VoidPtrPtrTy), "__cuda_gpubin_handle");
    CtorBuilder.CreateAlignedStore(RegisterFatbinCall, GpuBinaryHandle,
                                   CGM.getPointerAlign());

    // Call __cuda_register_globals(GpuBinaryHandle);
    if (RegisterGlobalsFunc)
      CtorBuilder.CreateCall(RegisterGlobalsFunc, RegisterFatbinCall);

    // Save GpuBinaryHandle so we can unregister it in destructor.
    GpuBinaryHandles.push_back(GpuBinaryHandle);
  }

  CtorBuilder.CreateRetVoid();
  return ModuleCtorFunc;
}

/// Creates a global destructor function that unregisters all GPU code blobs
/// registered by constructor.
/// \code
/// void __cuda_module_dtor(void*) {
///     __cudaUnregisterFatBinary(Handle0);
///     ...
///     __cudaUnregisterFatBinary(HandleN);
/// }
/// \endcode
llvm::Function *CGNVCUDARuntime::makeModuleDtorFunction() {
  // No need for destructor if we don't have handles to unregister.
  if (GpuBinaryHandles.empty())
    return nullptr;

  // void __cudaUnregisterFatBinary(void ** handle);
  llvm::Constant *UnregisterFatbinFunc = CGM.CreateRuntimeFunction(
      llvm::FunctionType::get(VoidTy, VoidPtrPtrTy, false),
      "__cudaUnregisterFatBinary");

  llvm::Function *ModuleDtorFunc = llvm::Function::Create(
      llvm::FunctionType::get(VoidTy, VoidPtrTy, false),
      llvm::GlobalValue::InternalLinkage, "__cuda_module_dtor", &TheModule);
  llvm::BasicBlock *DtorEntryBB =
      llvm::BasicBlock::Create(Context, "entry", ModuleDtorFunc);
  CGBuilderTy DtorBuilder(CGM, Context);
  DtorBuilder.SetInsertPoint(DtorEntryBB);

  for (llvm::GlobalVariable *GpuBinaryHandle : GpuBinaryHandles) {
    auto HandleValue =
      DtorBuilder.CreateAlignedLoad(GpuBinaryHandle, CGM.getPointerAlign());
    DtorBuilder.CreateCall(UnregisterFatbinFunc, HandleValue);
  }

  DtorBuilder.CreateRetVoid();
  return ModuleDtorFunc;
}

CGCUDARuntime *CodeGen::CreateNVCUDARuntime(CodeGenModule &CGM) {
  return new CGNVCUDARuntime(CGM);
}
