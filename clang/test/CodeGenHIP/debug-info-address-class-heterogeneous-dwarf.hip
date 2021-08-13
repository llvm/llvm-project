// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -emit-llvm -fcuda-is-device -debug-info-kind=limited -gheterogeneous-dwarf -o - %s | FileCheck %s

#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

// CHECK: @FileVarDevice = addrspace(1) externally_initialized global i32 0, align 4, !dbg.def ![[FILE_VAR_DEVICE_FRAGMENT:[0-9]+]]
__device__ int FileVarDevice;

// CHECK: @FileVarDeviceShared = addrspace(3) externally_initialized global i32 undef, align 4, !dbg.def ![[FILE_VAR_DEVICE_SHARED_FRAGMENT:[0-9]+]]
__device__ __shared__ int FileVarDeviceShared;

// CHECK: @FileVarDeviceConstant = addrspace(4) externally_initialized global i32 0, align 4, !dbg.def ![[FILE_VAR_DEVICE_CONSTANT_FRAGMENT:[0-9]+]]
__device__ __constant__ int FileVarDeviceConstant;

// CHECK: @_ZZ7kernel1iE13FuncVarShared = internal addrspace(3) global i32 undef, align 4, !dbg.def ![[FUNC_VAR_SHARED_FRAGMENT:[0-9]+]]

// CHECK: call void @llvm.dbg.def(metadata ![[ARG_LIFETIME:[0-9]+]], metadata i32* %Arg.addr.ascast)
__device__ void kernel1(int Arg) {

  __shared__ int FuncVarShared;

  // CHECK: call void @llvm.dbg.def(metadata ![[FUNC_VAR_LIFETIME:[0-9]+]], metadata i32* %FuncVar.ascast)
  int FuncVar;
}

// CHECK: !llvm.dbg.retainedNodes = !{![[FILE_VAR_DEVICE_LIFETIME:[0-9]+]], ![[FILE_VAR_DEVICE_SHARED_LIFETIME:[0-9]+]], ![[FILE_VAR_DEVICE_CONSTANT_LIFETIME:[0-9]+]], ![[FUNC_VAR_SHARED_LIFETIME:[0-9]+]]}

// CHECK: ![[FILE_VAR_DEVICE_FRAGMENT]] = distinct !DIFragment()
// CHECK: ![[FILE_VAR_DEVICE_SHARED_FRAGMENT]] = distinct !DIFragment()
// CHECK: ![[FILE_VAR_DEVICE_CONSTANT_FRAGMENT]] = distinct !DIFragment()
// CHECK: ![[FUNC_VAR_SHARED_FRAGMENT]] = distinct !DIFragment()

// CHECK: ![[FILE_VAR_DEVICE_LIFETIME]] = distinct !DILifetime(object: ![[FILE_VAR_DEVICE:[0-9]+]], location: !DIExpr(DIOpArg(0, i32 addrspace(1)*), DIOpDeref(i32)), argObjects: {![[FILE_VAR_DEVICE_FRAGMENT]]})
// CHECK: ![[FILE_VAR_DEVICE]] = distinct !DIGlobalVariable(name: "FileVarDevice",

// CHECK: ![[FILE_VAR_DEVICE_SHARED_LIFETIME]] = distinct !DILifetime(object: ![[FILE_VAR_DEVICE_SHARED:[0-9]+]], location: !DIExpr(DIOpArg(0, i32 addrspace(3)*), DIOpDeref(i32)), argObjects: {![[FILE_VAR_DEVICE_SHARED_FRAGMENT]]})
// CHECK: ![[FILE_VAR_DEVICE_SHARED]] = distinct !DIGlobalVariable(name: "FileVarDeviceShared",

// CHECK: ![[FILE_VAR_DEVICE_CONSTANT_LIFETIME]] = distinct !DILifetime(object: ![[FILE_VAR_DEVICE_CONSTANT:[0-9]+]], location: !DIExpr(DIOpArg(0, i32 addrspace(4)*), DIOpDeref(i32)), argObjects: {![[FILE_VAR_DEVICE_CONSTANT_FRAGMENT]]})
// CHECK: ![[FILE_VAR_DEVICE_CONSTANT]] = distinct !DIGlobalVariable(name: "FileVarDeviceConstant",

// CHECK: ![[FUNC_VAR_SHARED_LIFETIME]] = distinct !DILifetime(object: ![[FUNC_VAR_SHARED:[0-9]+]], location: !DIExpr(DIOpArg(0, i32 addrspace(3)*), DIOpDeref(i32)), argObjects: {![[FUNC_VAR_SHARED_FRAGMENT]]})
// CHECK: ![[FUNC_VAR_SHARED]] = distinct !DIGlobalVariable(name: "FuncVarShared",

// FIXME This should be in the private address space.
// CHECK: ![[ARG_LIFETIME]] = distinct !DILifetime(object: ![[ARG:[0-9]+]], location: !DIExpr(DIOpReferrer(i32*), DIOpDeref(i32)))
// CHECK: ![[ARG]] = !DILocalVariable(name: "Arg", arg: 1,

// FIXME This should be in the private address space.
// CHECK: ![[FUNC_VAR_LIFETIME]] = distinct !DILifetime(object: ![[FUNC_VAR:[0-9]+]], location: !DIExpr(DIOpReferrer(i32*), DIOpDeref(i32)))
// CHECK: ![[FUNC_VAR]] = !DILocalVariable(name: "FuncVar",
