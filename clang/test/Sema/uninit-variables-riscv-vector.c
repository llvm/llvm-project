// RUN: %clang_cc1 -triple riscv64 -fsyntax-only -Wuninitialized -fsyntax-only -target-feature +v %s -verify

#pragma clang riscv intrinsic vector

void test1(int *input, long vl) {
  __rvv_int32m1_t x, y, z, w, X; // expected-note {{variable 'x' is declared here}} expected-note {{variable 'y' is declared here}} expected-note {{variable 'w' is declared here}}  expected-note {{variable 'z' is declared here}}
  x = vxor_vv_i32m1(x,x, vl); // expected-warning {{variable 'x' is uninitialized when used here}}
  y = vxor_vv_i32m1(y,y, vl); // expected-warning {{variable 'y' is uninitialized when used here}}
  z = vxor_vv_i32m1(z,z, vl); // expected-warning {{variable 'z' is uninitialized when used here}}
  w = vxor_vv_i32m1(w,w, vl); // expected-warning {{variable 'w' is uninitialized when used here}}
  X = vle32_v_i32m1(&input[0], vl);
  X = vxor_vv_i32m1(X,X, vl); // no-warning
}

