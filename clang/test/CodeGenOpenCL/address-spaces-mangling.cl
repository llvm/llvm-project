// RUN: %clang_cc1 %s -ffake-address-space-map -faddress-space-map-mangling=yes -triple %itanium_abi_triple -emit-llvm -o - | FileCheck -check-prefixes="ASMANG,ASMANG10" %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -ffake-address-space-map -faddress-space-map-mangling=yes -triple %itanium_abi_triple -emit-llvm -o - | FileCheck -check-prefixes="ASMANG,ASMANG20" %s
// RUN: %clang_cc1 %s -ffake-address-space-map -faddress-space-map-mangling=no -triple %itanium_abi_triple -emit-llvm -o - | FileCheck -check-prefixes="NOASMANG,NOASMANG10" %s
// RUN: %clang_cc1 %s -cl-std=CL2.0 -ffake-address-space-map -faddress-space-map-mangling=no -triple %itanium_abi_triple -emit-llvm -o - | FileCheck -check-prefixes="NOASMANG,NOASMANG20" %s
// RUN: %clang_cc1 %s -cl-std=CL3.0 -cl-std=CL3.0 -cl-ext=+__opencl_c_generic_address_space -ffake-address-space-map -faddress-space-map-mangling=no -triple %itanium_abi_triple -emit-llvm -o - | FileCheck -check-prefixes="NOASMANG,NOASMANG20" %s
// RUN: %clang_cc1 %s -cl-std=CL3.0 -cl-ext=+__opencl_c_generic_address_space -ffake-address-space-map -faddress-space-map-mangling=yes -triple %itanium_abi_triple -emit-llvm -o - | FileCheck -check-prefixes="ASMANG,ASMANG20" %s

// We check that the address spaces are mangled the same in both version of OpenCL
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=CL2.0 -emit-llvm -o - | FileCheck -check-prefix=OCL-20 %s
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=CL1.2 -emit-llvm -o - | FileCheck -check-prefix=OCL-12 %s
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=CL3.0 -cl-ext=+__opencl_c_generic_address_space -emit-llvm -o - | FileCheck -check-prefix=OCL-20 %s
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=CL3.0 -cl-ext=-__opencl_c_generic_address_space -emit-llvm -o - | FileCheck -check-prefix=OCL-12 %s

// We can't name this f as private is equivalent to default
// no specifier given address space so we get multiple definition
// warnings, but we do want it for comparison purposes.
__attribute__((overloadable))
void ff(int *arg) { }
// ASMANG10: @_Z2ffPi
// ASMANG20: @_Z2ffPU3AS4i
// NOASMANG10: @_Z2ffPU9CLprivatei
// NOASMANG20: @_Z2ffPU9CLgenerici
// OCL-20-DAG: @_Z2ffPU3AS4i
// OCL-12-DAG: @_Z2ffPi

__attribute__((overloadable))
void f(private int *arg) { }
// ASMANG: @_Z1fPi
// NOASMANG: @_Z1fPU9CLprivatei
// OCL-20-DAG: @_Z1fPi
// OCL-12-DAG: @_Z1fPi

__attribute__((overloadable))
void f(global int *arg) { }
// ASMANG: @_Z1fPU3AS1i
// NOASMANG: @_Z1fPU8CLglobali
// OCL-20-DAG: @_Z1fPU3AS1i
// OCL-12-DAG: @_Z1fPU3AS1i

__attribute__((overloadable))
void f(local int *arg) { }
// ASMANG: @_Z1fPU3AS3i
// NOASMANG: @_Z1fPU7CLlocali
// OCL-20-DAG: @_Z1fPU3AS3i
// OCL-12-DAG: @_Z1fPU3AS3i

__attribute__((overloadable))
void f(constant int *arg) { }
// ASMANG: @_Z1fPU3AS2i
// NOASMANG: @_Z1fPU10CLconstanti
// OCL-20-DAG: @_Z1fPU3AS2i
// OCL-12-DAG: @_Z1fPU3AS2i

#if (__OPENCL_C_VERSION__ == 200) || defined(__opencl_c_generic_address_space)
__attribute__((overloadable))
void f(generic int *arg) { }
// ASMANG20: @_Z1fPU3AS4i
// NOASMANG20: @_Z1fPU9CLgenerici
// OCL-20-DAG: @_Z1fPU3AS4i
#endif
