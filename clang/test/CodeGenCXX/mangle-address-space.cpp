// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s --check-prefixes=CHECK,CHECKNOOCL
// RUN: %clang_cc1 -emit-llvm -triple x86_64-windows-msvc -o - %s | FileCheck %s --check-prefixes=WIN,WINNOOCL
// RUN: %clang_cc1 -cl-std=clc++ -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s --check-prefixes=CHECK,CHECKOCL
// RUN: %clang_cc1 -cl-std=clc++ -emit-llvm -triple x86_64-windows-msvc -o - %s | FileCheck %s --check-prefixes=WIN,WINOCL

// CHECKNOOCL-LABEL: define {{.*}}void @_Z2f0Pc
// WINNOOCL-LABEL: define {{.*}}void @"?f0@@YAXPEAD@Z"
// CHECKOCL-LABEL: define {{.*}}void @_Z2f0PU9CLgenericc
// WINOCL-LABEL: define {{.*}}void @"?f0@@YAXPEAU?$_ASCLgeneric@$$CAD@__clang@@@Z" 
void f0(char *p) { }
// CHECK-LABEL: define {{.*}}void @_Z2f0PU3AS1c
// WIN-LABEL: define {{.*}}void @"?f0@@YAXPEAU?$_AS@$00$$CAD@__clang@@@Z"
void f0(char __attribute__((address_space(1))) *p) { }

struct OpaqueType;
typedef OpaqueType __attribute__((address_space(100))) * OpaqueTypePtr;

// CHECK-LABEL: define {{.*}}void @_Z2f0PU5AS10010OpaqueType
// WIN-LABEL: define {{.*}}void @"?f0@@YAXPEAU?$_AS@$0GE@$$CAUOpaqueType@@@__clang@@@Z"
void f0(OpaqueTypePtr) { }

// CHECK-LABEL: define {{.*}}void @_Z2f1PU3AS1Kc
// WIN-LABEL: define {{.*}}void @"?f1@@YAXPEAU?$_AS@$00$$CBD@__clang@@@Z"
void f1(char __attribute__((address_space(1))) const *p) {}

// Ensure we can do return values, which change in MS mode.
// CHECK-LABEL: define {{.*}}ptr addrspace(1) @_Z2f1PU3AS2Kc
// WIN-LABEL: define {{.*}}ptr addrspace(1) @"?f1@@YAPEAU?$_AS@$00$$CAM@__clang@@PEAU?$_AS@$01$$CBD@2@@Z"
__attribute__((address_space(1))) float *f1(char __attribute__((address_space(2))) const *p) { return 0;}

#if !defined(__OPENCL_CPP_VERSION__)
// Return value of address space without a pointer is invalid in opencl.
// Ensure we skip return values, since non-pointers aren't supposed to have an AS.
// CHECKNOOCL-LABEL: define {{.*}}float @_Z2f2PU3AS2Kc
// WINNOOCL-LABEL: define {{.*}}float @"?f2@@YA?AMQEAU?$_AS@$01$$CBD@__clang@@@Z"
__attribute__((address_space(1))) float f2(char __attribute__((address_space(2))) const * const p) { return 0;}
#endif

#ifdef __OPENCL_CPP_VERSION__
// CHECKOCL-LABEL: define {{.*}}void @_Z6ocl_f0PU9CLprivatec
// WINOCL-LABEL: define {{.*}}void @"?ocl_f0@@YAXPEAU?$_ASCLprivate@$$CAD@__clang@@@Z"
void ocl_f0(char __private *p) { }

struct ocl_OpaqueType;
typedef ocl_OpaqueType __global * ocl_OpaqueTypePtr;
typedef ocl_OpaqueType __attribute__((opencl_global_host)) * ocl_OpaqueTypePtrH;
typedef ocl_OpaqueType
    __attribute__((opencl_global_device)) *
    ocl_OpaqueTypePtrD;

// CHECKOCL-LABEL: define {{.*}}void @_Z6ocl_f0PU8CLglobal14ocl_OpaqueType
// WINOCL-LABEL: define {{.*}}void @"?ocl_f0@@YAXPEAU?$_ASCLglobal@$$CAUocl_OpaqueType@@@__clang@@@Z"
void ocl_f0(ocl_OpaqueTypePtr) { }

// CHECKOCL-LABEL: define {{.*}}void @_Z6ocl_f1PU10CLconstantKc
// WINOCL-LABEL: define {{.*}}void @"?ocl_f1@@YAXPEAU?$_ASCLconstant@$$CBD@__clang@@@Z"
void ocl_f1(char __constant const *p) {}

// Ensure we can do return values, which change in MS mode.
// CHECKOCL-LABEL: define {{.*}}ptr @_Z6ocl_f1PU9CLgenericKc
// WINOCL-LABEL: define {{.*}}ptr @"?ocl_f1@@YAPEAU?$_ASCLconstant@$$CAM@__clang@@PEAU?$_ASCLgeneric@$$CBD@2@@Z"
__constant float *ocl_f1(char __generic const *p) { return 0;}

// Ensure we skip return values, since non-pointers aren't supposed to have an AS.
// CHECKOCL-LABEL: define {{.*}}ptr @_Z6ocl_f2PU9CLgenericKc
// WINOCL-LABEL: define {{.*}}ptr @"?ocl_f2@@YAPEAU?$_ASCLgeneric@$$CAM@__clang@@QEAU?$_ASCLgeneric@$$CBD@2@@Z"
__generic float *ocl_f2(__generic char const * const p) { return 0;}

// CHECKOCL-LABEL: define {{.*}}void @_Z6ocl_f3PU6CLhost14ocl_OpaqueType
// WINOCL-LABEL: define {{.*}}void @"?ocl_f3@@YAXPEAU?$_ASCLhost@$$CAUocl_OpaqueType@@@__clang@@@Z"
void ocl_f3(ocl_OpaqueTypePtrH) {}

// CHECKOCL-LABEL: define {{.*}}void @_Z6ocl_f4PU8CLdevice14ocl_OpaqueType
// WINOCL-LABEL: define {{.*}}void @"?ocl_f4@@YAXPEAU?$_ASCLdevice@$$CAUocl_OpaqueType@@@__clang@@@Z"
void ocl_f4(ocl_OpaqueTypePtrD) {}
#endif
