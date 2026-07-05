// RUN: %clang_cc1 -fexceptions -fcxx-exceptions -std=c++1z -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s

// CHECK-LABEL: define {{.*}} @_Z11builtin_newm(
// CHECK: call {{.*}} @_Znwm(
void *builtin_new(unsigned long n) { return __builtin_operator_new(n); }

// CHECK-LABEL: define {{.*}} @_Z14builtin_deletePv(
// CHECK: call {{.*}} @_ZdlPv(
void builtin_delete(void *p) { return __builtin_operator_delete(p); }
