// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -verify %s -o -

void test_stdc_trailing_zeros_undef_cast(unsigned long long x) {
  // expected-error@+1 {{ClangIR code gen Not Yet Implemented: unimplemented builtin call: __builtin_stdc_trailing_zeros}}
  int cnt = __builtin_stdc_trailing_zeros(x);
  (void)cnt;
}
