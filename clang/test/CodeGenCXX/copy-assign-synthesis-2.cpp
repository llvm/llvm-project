// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s
struct A {};
A& (A::*x)(const A&) = &A::operator=;
// CHECK-LABEL: define linkonce_odr {{.*}}ptr @_ZN1AaSERKS_
