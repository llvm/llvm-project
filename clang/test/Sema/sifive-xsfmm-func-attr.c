// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +xsfmmbase -Wno-error=implicit-function-declaration -fsyntax-only -verify %s

#include <riscv_vector.h>

int xsfmm_in_callee(void) __riscv_in("xsfmm");
int xsfmm_out_callee(void) __riscv_out("xsfmm");
int xsfmm_inout_callee(void) __riscv_inout("xsfmm");
int xsfmm_preserves_callee(void) __riscv_preserves("xsfmm");
int xsfmm_new_callee(void) __riscv_new("xsfmm");

void valid_new(void) {
  xsfmm_new_callee();
}

void valid_preserves_in_in(void) __riscv_in("xsfmm") {
  xsfmm_preserves_callee();
}

void valid_in_in_in(void) __riscv_in("xsfmm") {
  xsfmm_in_callee();
}

void valid_in_in_out(void) __riscv_out("xsfmm") {
  xsfmm_in_callee();
}

void valid_out_in_out(void) __riscv_out("xsfmm") {
  xsfmm_out_callee();
}

void valid_preserves_in_out(void) __riscv_out("xsfmm") {
  xsfmm_preserves_callee();
}

void valid_preserves_in_preserves(void) __riscv_preserves("xsfmm") {
  xsfmm_preserves_callee();
}

vint32m1_t valid_in_intrinsic(vint32m1_t v, unsigned vl) __riscv_in("xsfmm") {
  unsigned avl = __riscv_vsetvl_e32m1(vl);
  return __riscv_vadd(v, v, avl);
}

vint32m1_t valid_out_intrinsic(vint32m1_t v, unsigned vl) __riscv_out("xsfmm") {
  unsigned avl = __riscv_vsetvl_e32m1(vl);
  return __riscv_vadd(v, v, avl);
}

vint32m1_t valid_inout_intrinsic(vint32m1_t v, unsigned vl) __riscv_inout("xsfmm") {
  unsigned avl = __riscv_vsetvl_e32m1(vl);
  return __riscv_vadd(v, v, avl);
}

vint32m1_t valid_preserves_intrinsic(vint32m1_t v, unsigned vl) __riscv_preserves("xsfmm") {
  unsigned avl = __riscv_vsetvl_e32m1(vl);
  return __riscv_vadd(v, v, avl);
}

vint32m1_t valid_new_intrinsic(vint32m1_t v, unsigned vl) __riscv_new("xsfmm") {
  unsigned avl = __riscv_vsetvl_e32m1(vl);
  return __riscv_vadd(v, v, avl);
}

void invalid_mutual_exclusive1(void) __riscv_in("xsfmm") __riscv_out("xsfmm") { // expected-error {{mutually exclusive attributes for state 'xsfmm'}}
}

void invalid_mutual_exclusive2(void) __riscv_in("xsfmm") __riscv_inout("xsfmm") { // expected-error {{mutually exclusive attributes for state 'xsfmm'}}
}

void invalid_mutual_exclusive3(void) __riscv_in("xsfmm") __riscv_preserves("xsfmm") { // expected-error {{mutually exclusive attributes for state 'xsfmm'}}
}

void invalid_mutual_exclusive4(void) __riscv_in("xsfmm") __riscv_new("xsfmm") { // expected-error {{mutually exclusive attributes for state 'xsfmm'}}
}

void invalid_mutual_exclusive5(void) __riscv_out("xsfmm") __riscv_inout("xsfmm") { // expected-error {{mutually exclusive attributes for state 'xsfmm'}}
}

void invalid_mutual_exclusive6(void) __riscv_out("xsfmm") __riscv_preserves("xsfmm") { // expected-error {{mutually exclusive attributes for state 'xsfmm'}}
}

void invalid_mutual_exclusive7(void) __riscv_out("xsfmm") __riscv_new("xsfmm") { // expected-error {{mutually exclusive attributes for state 'xsfmm'}}
}

void invalid_mutual_exclusive8(void) __riscv_inout("xsfmm") __riscv_preserves("xsfmm") { // expected-error {{mutually exclusive attributes for state 'xsfmm'}}
}

void invalid_mutual_exclusive9(void) __riscv_inout("xsfmm") __riscv_new("xsfmm") { // expected-error {{mutually exclusive attributes for state 'xsfmm'}}
}

void invalid_mutual_exclusive10(void) __riscv_preserves("xsfmm") __riscv_new("xsfmm") { // expected-error {{mutually exclusive attributes for state 'xsfmm'}}
}

void invalid_in_in_preserves(void) __riscv_preserves("xsfmm") {
  xsfmm_in_callee(); // expected-error {{conflicting attribute. Description: Function with __riscv_preserves can only call __riscv_preserves function.}}
}

void invalid_out_in_preserves(void) __riscv_preserves("xsfmm") {
  xsfmm_out_callee(); // expected-error {{conflicting attribute. Description: Function with __riscv_preserves can only call __riscv_preserves function.}}
}

void invalid_inout_in_preserves(void) __riscv_preserves("xsfmm") {
  xsfmm_inout_callee(); // expected-error {{conflicting attribute. Description: Function with __riscv_preserves can only call __riscv_preserves function.}}
}

void invalid_new_in_preserves(void) __riscv_preserves("xsfmm") {
  xsfmm_new_callee(); // expected-error {{conflicting attribute. Description: Function with __riscv_preserves can only call __riscv_preserves function.}}
}

void invalid_out_in_in(void) __riscv_in("xsfmm") {
  xsfmm_out_callee(); // expected-error {{conflicting attribute. Description: Function with __riscv_in can only call __riscv_in and __riscv_preserves function.}}
}

void invalid_inout_in_in(void) __riscv_in("xsfmm") {
  xsfmm_inout_callee(); // expected-error {{conflicting attribute. Description: Function with __riscv_in can only call __riscv_in and __riscv_preserves function.}}
}

void invalid_new_in_in(void) __riscv_in("xsfmm") {
  xsfmm_new_callee(); // expected-error {{conflicting attribute. Description: Function with __riscv_in can only call __riscv_in and __riscv_preserves function.}}
}

void invalid_inout_in_out(void) __riscv_out("xsfmm") {
  xsfmm_inout_callee(); // expected-error {{conflicting attribute. Description: Function with __riscv_out can only call __riscv_in, __riscv_out and __riscv_preserves function.}}
}

void invalid_new_in_out(void) __riscv_out("xsfmm") {
  xsfmm_new_callee(); // expected-error {{conflicting attribute. Description: Function with __riscv_out can only call __riscv_in, __riscv_out and __riscv_preserves function.}}
}

void invalid_new_in_new(void) __riscv_new("xsfmm") {
  xsfmm_new_callee(); // expected-error {{conflicting attribute. Description: __riscv_new and non-attributed function can not be called in attributed function.}}
}

void invalid_lib_function_intrinsic(void) __riscv_in("xsfmm") {
  __builtin_memcpy(NULL, NULL, 0); // expected-error {{conflicting attribute. Description: Function with __riscv_in can only call __riscv_in and __riscv_preserves function.}}
}

void invalid_predefined_lib_function(void) __riscv_in("xsfmm") {
  memcpy(NULL, NULL, 0); // expected-error {{conflicting attribute. Description: Function with __riscv_in can only call __riscv_in and __riscv_preserves function.}}
  // expected-warning@-1 {{call to undeclared library function 'memcpy' with type 'void *(void *, const void *, __size_t)' (aka 'void *(void *, const void *, unsigned long)'); ISO C99 and later do not support implicit function declarations}}
  // expected-note@-2 {{include the header <string.h> or explicitly provide a declaration for 'memcpy'}}
}
