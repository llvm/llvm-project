// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -verify -DBITFIELD_COUNT %s -o -
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -verify -DCONDITIONAL_BASE %s -o -

#ifdef BITFIELD_COUNT
struct bf {
  int count : 8;
  int array[] __attribute__((counted_by(count)));
};

void test_bitfield_count(struct bf *p, int size) {
  *__builtin_counted_by_ref(p->array) = size; // expected-error {{ClangIR code gen Not Yet Implemented: __builtin_counted_by_ref: bit-field count}}
}
#endif

#ifdef CONDITIONAL_BASE
struct s {
  int count;
  int array[] __attribute__((counted_by(count)));
};

void test_conditional_base(int cond, struct s *p, struct s *q, int size) {
  *__builtin_counted_by_ref((cond ? p : q)->array) = size; // expected-error {{ClangIR code gen Not Yet Implemented: __builtin_counted_by_ref: unsupported base expression}}
}
#endif
