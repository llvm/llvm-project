// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cp %S/Inputs/const-correctness/correctness.h %t/correctness.h
// RUN: %check_clang_tidy %s misc-const-correctness %t/temp -- \
// RUN:   -- -I%t -fno-delayed-template-parsing
// RUN: diff %t/correctness.h %S/Inputs/const-correctness/correctness-fixed.h

#include "correctness.h"

// CHECK-MESSAGES: :[[@LINE+1]]:26: warning: variable 's' of type 'S &' can be declared 'const'
void func_with_ref_param(S& s) {
  // CHECK-FIXES: void func_with_ref_param(S const& s) {
  s.method();
}

// CHECK-MESSAGES: :[[@LINE+1]]:31: warning: variable 'readonly' of type 'S &' can be declared 'const'
void func_mixed_params(int x, S& readonly, S& mutated) {
  // CHECK-FIXES: void func_mixed_params(int x, S const& readonly, S& mutated) {
  readonly.method();
  mutated.value = x;
}

// CHECK-MESSAGES: :[[@LINE+1]]:22: warning: variable 's' of type 'S &' can be declared 'const'
void multi_decl_func(S& s) {
  // CHECK-FIXES: void multi_decl_func(S const& s) {
  s.method();
}
