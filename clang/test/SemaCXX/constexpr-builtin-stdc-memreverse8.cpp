// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=ref,both -std=c++20 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify=expected,both -std=c++20 %s -fexperimental-new-constant-interpreter

namespace test_noop {

// N=0: no bytes touched.
constexpr bool test_n0() {
  unsigned char buf[4] = {0x12, 0x34, 0x56, 0x78};
  __builtin_stdc_memreverse8(0, buf);
  return buf[0] == 0x12;
}
static_assert(test_n0(), "");

// N=1: single byte is its own reverse.
constexpr bool test_n1() {
  unsigned char buf[4] = {0x12, 0x34, 0x56, 0x78};
  __builtin_stdc_memreverse8(1, buf);
  return buf[0] == 0x12;
}
static_assert(test_n1(), "");

// N=0 with a one-past-the-end pointer: valid since no bytes are accessed.
constexpr bool test_n0_one_past_end() {
  unsigned char buf[2] = {0x12, 0x34};
  __builtin_stdc_memreverse8(0, buf + 2);
  return buf[0] == 0x12 && buf[1] == 0x34;
}
static_assert(test_n0_one_past_end(), "");

} // namespace test_noop

namespace test_basic {

// N=2: swap two bytes.
constexpr bool test_n2() {
  unsigned char buf[2] = {0x12, 0x34};
  __builtin_stdc_memreverse8(2, buf);
  return buf[0] == 0x34 && buf[1] == 0x12;
}
static_assert(test_n2(), "");

// N=4: full 4-byte reversal.
constexpr bool test_n4() {
  unsigned char buf[4] = {0x12, 0x34, 0x56, 0x78};
  __builtin_stdc_memreverse8(4, buf);
  return buf[0] == 0x78 && buf[1] == 0x56 &&
         buf[2] == 0x34 && buf[3] == 0x12;
}
static_assert(test_n4(), "");

// N=8: 8-byte reversal.
constexpr bool test_n8() {
  unsigned char buf[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
  __builtin_stdc_memreverse8(8, buf);
  return buf[0] == 0x08 && buf[1] == 0x07 && buf[2] == 0x06 &&
         buf[3] == 0x05 && buf[4] == 0x04 && buf[5] == 0x03 &&
         buf[6] == 0x02 && buf[7] == 0x01;
}
static_assert(test_n8(), "");

} // namespace test_basic

namespace test_odd_length {

// N=3: odd-length reversal, middle byte unchanged.
constexpr bool test_n3() {
  unsigned char buf[3] = {0xAA, 0xBB, 0xCC};
  __builtin_stdc_memreverse8(3, buf);
  return buf[0] == 0xCC && buf[1] == 0xBB && buf[2] == 0xAA;
}
static_assert(test_n3(), "");

// N=5: odd-length, middle byte stays.
constexpr bool test_n5() {
  unsigned char buf[5] = {0x01, 0x02, 0x03, 0x04, 0x05};
  __builtin_stdc_memreverse8(5, buf);
  return buf[0] == 0x05 && buf[1] == 0x04 && buf[2] == 0x03 &&
         buf[3] == 0x02 && buf[4] == 0x01;
}
static_assert(test_n5(), "");

} // namespace test_odd_length

namespace test_idempotent {

// All same bytes: reversal is a no-op.
constexpr bool test_all_same() {
  unsigned char buf[4] = {0xAB, 0xAB, 0xAB, 0xAB};
  __builtin_stdc_memreverse8(4, buf);
  return buf[0] == 0xAB && buf[1] == 0xAB &&
         buf[2] == 0xAB && buf[3] == 0xAB;
}
static_assert(test_all_same(), "");

// Palindrome: reversal produces same sequence.
constexpr bool test_palindrome() {
  unsigned char buf[4] = {0x12, 0x34, 0x34, 0x12};
  __builtin_stdc_memreverse8(4, buf);
  return buf[0] == 0x12 && buf[1] == 0x34 &&
         buf[2] == 0x34 && buf[3] == 0x12;
}
static_assert(test_palindrome(), "");

} // namespace test_idempotent

namespace test_double_reverse {

// Reversing twice restores the original.
constexpr bool test_round_trip_4() {
  unsigned char buf[4] = {0x12, 0x34, 0x56, 0x78};
  __builtin_stdc_memreverse8(4, buf);
  __builtin_stdc_memreverse8(4, buf);
  return buf[0] == 0x12 && buf[1] == 0x34 &&
         buf[2] == 0x56 && buf[3] == 0x78;
}
static_assert(test_round_trip_4(), "");

constexpr bool test_round_trip_3() {
  unsigned char buf[3] = {0xDE, 0xAD, 0xBE};
  __builtin_stdc_memreverse8(3, buf);
  __builtin_stdc_memreverse8(3, buf);
  return buf[0] == 0xDE && buf[1] == 0xAD && buf[2] == 0xBE;
}
static_assert(test_round_trip_3(), "");

} // namespace test_double_reverse

namespace test_partial {

// Reverse only part of a larger buffer; rest is untouched.
constexpr bool test_partial_reverse() {
  unsigned char buf[6] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06};
  __builtin_stdc_memreverse8(4, buf);
  return buf[0] == 0x04 && buf[1] == 0x03 && buf[2] == 0x02 &&
         buf[3] == 0x01 && buf[4] == 0x05 && buf[5] == 0x06;
}
static_assert(test_partial_reverse(), "");

// ptr points into the middle of a larger buffer; bytes outside the range are untouched.
constexpr bool test_mid_array() {
  unsigned char buf[6] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06};
  __builtin_stdc_memreverse8(4, buf + 1);
  return buf[0] == 0x01 && buf[1] == 0x05 && buf[2] == 0x04 &&
         buf[3] == 0x03 && buf[4] == 0x02 && buf[5] == 0x06;
}
static_assert(test_mid_array(), "");

// N=1 with a pointer-to-scalar (not an array): single byte is a no-op.
constexpr bool test_scalar_n1() {
  unsigned char c = 0x42;
  __builtin_stdc_memreverse8(1, &c);
  return c == 0x42;
}
static_assert(test_scalar_n1(), "");

// N=0 with a pointer-to-scalar: no-op.
constexpr bool test_scalar_n0() {
  unsigned char c = 0x42;
  __builtin_stdc_memreverse8(0, &c);
  return c == 0x42;
}
static_assert(test_scalar_n0(), "");

} // namespace test_partial

namespace test_side_effects {

// n++ passes n=4 to the reversal, then increments n to 5.
constexpr bool test_n_postincrement() {
  unsigned char buf[4] = {0x01, 0x02, 0x03, 0x04};
  __SIZE_TYPE__ n = 4;
  __builtin_stdc_memreverse8(n++, buf);
  return n == 5 &&
         buf[0] == 0x04 && buf[1] == 0x03 && buf[2] == 0x02 && buf[3] == 0x01;
}
static_assert(test_n_postincrement(), "");

// ++n increments n to 4 first, then passes n=4 to the reversal.
constexpr bool test_n_preincrement() {
  unsigned char buf[4] = {0x01, 0x02, 0x03, 0x04};
  __SIZE_TYPE__ n = 3;
  __builtin_stdc_memreverse8(++n, buf);
  return n == 4 &&
         buf[0] == 0x04 && buf[1] == 0x03 && buf[2] == 0x02 && buf[3] == 0x01;
}
static_assert(test_n_preincrement(), "");

// p++ passes buf to the reversal, then advances p.
constexpr bool test_p_postincrement() {
  unsigned char buf[4] = {0x01, 0x02, 0x03, 0x04};
  unsigned char *p = buf;
  __builtin_stdc_memreverse8(4, p++);
  return buf[0] == 0x04 && buf[1] == 0x03 && buf[2] == 0x02 && buf[3] == 0x01 &&
         p == buf + 1;
}
static_assert(test_p_postincrement(), "");

// Both arguments have side effects; each is evaluated once before the call.
constexpr bool test_both_postincrement() {
  unsigned char buf[5] = {0x01, 0x02, 0x03, 0x04, 0x05};
  unsigned char *p = buf;
  __SIZE_TYPE__ n = 4;
  __builtin_stdc_memreverse8(n++, p++);
  return n == 5 && p == buf + 1 &&
         buf[0] == 0x04 && buf[1] == 0x03 && buf[2] == 0x02 && buf[3] == 0x01 &&
         buf[4] == 0x05;
}
static_assert(test_both_postincrement(), "");

} // namespace test_side_effects

namespace test_side_effects_zero_or_one {

// Side effects in the second argument must be evaluated even if N == 0.
constexpr bool test_n0_side_effect() {
  unsigned char buf[4] = {0x12, 0x34, 0x56, 0x78};
  unsigned char *p = buf;
  __builtin_stdc_memreverse8(0, p++);
  return p == buf + 1;
}
static_assert(test_n0_side_effect(), "");

// Side effects in the second argument must be evaluated even if N == 1.
constexpr bool test_n1_side_effect() {
  unsigned char buf[4] = {0x12, 0x34, 0x56, 0x78};
  unsigned char *p = buf;
  __builtin_stdc_memreverse8(1, p++);
  return p == buf + 1;
}
static_assert(test_n1_side_effect(), "");

} // namespace test_side_effects_zero_or_one

namespace test_negative {

// N exceeds the array size: out-of-bounds access.
constexpr bool test_oob() { // both-error{{constexpr function never produces a constant expression}}
  unsigned char buf[2] = {0x01, 0x02};
  __builtin_stdc_memreverse8(4, buf); // both-note 2{{cannot refer to element 3 of array of 2 elements in a constant expression}}
  return true;
}
static_assert(test_oob(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_oob()'}}

// Null pointer: assignment through null is not allowed in a constant expression.
constexpr bool test_null_ptr() { // both-error{{constexpr function never produces a constant expression}}
  __builtin_stdc_memreverse8(4, (unsigned char *)0); // both-warning{{null passed to a callee that requires a non-null argument}} both-note 2{{assignment to dereferenced null pointer is not allowed in a constant expression}}
  return true;
}
static_assert(test_null_ptr(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_null_ptr()'}}

// N=0 but null pointer passed.
constexpr bool test_null_ptr_n0() { // both-error{{constexpr function never produces a constant expression}}
  __builtin_stdc_memreverse8(0, (unsigned char *)0); // both-warning{{null passed to a callee that requires a non-null argument}} both-note 2{{assignment to dereferenced null pointer is not allowed in a constant expression}}
  return true;
}
static_assert(test_null_ptr_n0(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_null_ptr_n0()'}}

// N=1 but null pointer passed.
constexpr bool test_null_ptr_n1() { // both-error{{constexpr function never produces a constant expression}}
  __builtin_stdc_memreverse8(1, (unsigned char *)0); // both-warning{{null passed to a callee that requires a non-null argument}} both-note 2{{assignment to dereferenced null pointer is not allowed in a constant expression}}
  return true;
}
static_assert(test_null_ptr_n1(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_null_ptr_n1()'}}

// N=1 but pointer is one-past-the-end.
constexpr bool test_oob_n1() { // both-error{{constexpr function never produces a constant expression}}
  unsigned char buf[2] = {0x01, 0x02};
  __builtin_stdc_memreverse8(1, buf + 2); // both-note 2{{cannot refer to element 2 of array of 2 elements in a constant expression}}
  return true;
}
static_assert(test_oob_n1(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_oob_n1()'}}

// Swapping uninitialized memory.
constexpr bool test_uninit() {
  unsigned char buf[2]; // both-note {{declared here}}
  __builtin_stdc_memreverse8(2, buf); // both-note {{read of uninitialized object is not allowed in a constant expression}}
  return true;
}
static_assert(test_uninit(), ""); // both-error{{not an integral constant expression}} both-note{{in call to 'test_uninit()'}}

} // namespace test_negative
