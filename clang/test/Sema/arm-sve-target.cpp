// RUN: %clang_cc1 -fsyntax-only -verify -DNONEON -std=c++11 -triple aarch64 %s

// A target without sve should not be able to use sve types.

void test_var() {
  __SVFloat32_t x; // expected-error {{SVE vector type '__SVFloat32_t' cannot be used in a target without sve}}
}

__attribute__((target("sve")))
void test_var_target() {
  __SVFloat32_t x;
}

__attribute__((target("sve2")))
void test_var_target2() {
  __SVFloat32_t x;
}

__attribute__((target("sve2-bitperm")))
void test_var_target3() {
  __SVFloat32_t x;
}

__SVFloat32_t other_ret();
__SVFloat32_t test_ret() { // expected-error {{SVE vector type '__SVFloat32_t' cannot be used in a target without sve}}
  return other_ret(); // expected-error {{SVE vector type '__SVFloat32_t' cannot be used in a target without sve}}
}

__attribute__((target("sve")))
__SVFloat32_t test_ret_target() {
  return other_ret();
}

void test_arg(__SVFloat32_t arg) { // expected-error {{SVE vector type '__SVFloat32_t' cannot be used in a target without sve}}
}

__attribute__((target("sve")))
void test_arg_target(__SVFloat32_t arg) {
}

__clang_svint32x4_t test4x() { // expected-error {{SVE vector type '__clang_svint32x4_t' cannot be used in a target without sve}}
  __clang_svint32x4_t x; // expected-error {{SVE vector type '__clang_svint32x4_t' cannot be used in a target without sve}}
  return x;
}

__attribute__((target("sve")))
__clang_svint32x4_t test4x_target() {
  __clang_svint32x4_t x;
  return x;
}

// Pointers are still valid to pass around.
void foo(__SVFloat32_t *&ptrA, __SVFloat32_t* &ptrB) {
    ptrA = ptrB;
}

__SVFloat32_t* foo(int x, __SVFloat32_t *ptrA) {
    return ptrA;
}

