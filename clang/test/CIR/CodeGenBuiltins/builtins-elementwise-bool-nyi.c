// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir %s -verify -emit-cir -o -

typedef _Bool vbool4 __attribute__((ext_vector_type(4)));

void test_bool_sat(_Bool a, _Bool b, vbool4 va, vbool4 vb) {
  // expected-error@+1 {{ClangIR code gen Not Yet Implemented: saturating add/sub on a boolean operand}}
  (void)__builtin_elementwise_add_sat(a, b);
  // expected-error@+1 {{ClangIR code gen Not Yet Implemented: saturating add/sub on a boolean operand}}
  (void)__builtin_elementwise_sub_sat(a, b);
  // expected-error@+1 {{ClangIR code gen Not Yet Implemented: saturating add/sub on a boolean operand}}
  (void)__builtin_elementwise_add_sat(va, vb);
  // expected-error@+1 {{ClangIR code gen Not Yet Implemented: saturating add/sub on a boolean operand}}
  (void)__builtin_elementwise_sub_sat(va, vb);
}
