// RUN: %clang_cc1 %s -ffake-address-space-map -faddress-space-map-mangling=yes -triple %itanium_abi_triple -emit-llvm -o - | FileCheck -check-prefix=ASMANG %s
// RUN: %clang_cc1 %s -ffake-address-space-map -faddress-space-map-mangling=no -triple %itanium_abi_triple -emit-llvm -o - | FileCheck -check-prefix=NOASMANG %s

// We check that the address spaces are mangled the same in both version of OpenCL
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=CL2.0 -emit-llvm -o - | FileCheck -check-prefix=OCL-20 %s
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=CL1.2 -emit-llvm -o - | FileCheck -check-prefix=OCL-12 %s

// We can't name this f as private is equivalent to default
// no specifier given address space so we get multiple definition
// warnings, but we do want it for comparison purposes.
__attribute__((overloadable))
void ff(int *arg) { }
// ASMANG: @_Z2ffPi
// NOASMANG: @_Z2ffPi
// OCL-20-DAG: @_Z2ffPU3AS4i
// OCL-12-DAG: @_Z2ffPi

__attribute__((overloadable))
void f(private int *arg) { }
// ASMANG: @_Z1fPi
// NOASMANG: @_Z1fPi
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
