// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -o - %s | FileCheck %s

// CHECK-LABEL: define {{.*}}void @_Z2f0Pc
void f0(char *p) { }
// CHECK-LABEL: define {{.*}}void @_Z2f0PU3AS1c
void f0(char __attribute__((address_space(1))) *p) { }

struct OpaqueType;
typedef OpaqueType __attribute__((address_space(100))) * OpaqueTypePtr;

// CHECK-LABEL: define {{.*}}void @_Z2f0PU5AS10010OpaqueType
void f0(OpaqueTypePtr) { }

// CHECK-LABEL: define {{.*}}void @_Z2f1PU3AS1Kc
void f1(char __attribute__((address_space(1))) const *p) {}