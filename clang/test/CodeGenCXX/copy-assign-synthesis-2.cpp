// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
struct A {};
A& (A::*x)(const A&) = &A::operator=;
// CHECK: define linkonce_odr %struct.A* @_ZN1AaSERKS_
