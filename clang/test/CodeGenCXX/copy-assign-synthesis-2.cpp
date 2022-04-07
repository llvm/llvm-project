// RUN: %clang_cc1 -no-opaque-pointers -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s
struct A {};
A& (A::*x)(const A&) = &A::operator=;
// CHECK-LABEL: define linkonce_odr {{.*}}%struct.A* @_ZN1AaSERKS_
