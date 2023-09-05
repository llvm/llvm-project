// RUN: %clang_analyze_cc1 -analyzer-checker=core.BitwiseShift \
// RUN:    -analyzer-config core.BitwiseShift:Pedantic=true \
// RUN:    -analyzer-output=text -verify=expected,c \
// RUN:    -triple x86_64-pc-linux-gnu -x c %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core.BitwiseShift \
// RUN:    -analyzer-config core.BitwiseShift:Pedantic=true \
// RUN:    -analyzer-output=text -verify=expected,cxx \
// RUN:    -triple x86_64-pc-linux-gnu -x c++ -std=c++14 %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow

// This test file verifies the pedantic mode of the BitwiseShift checker, which
// also reports issues that are undefined behavior (according to the standard,
// under C and in C++ before C++20), but would be accepted by many compilers.

// TEST NEGATIVE LEFT OPERAND
//===----------------------------------------------------------------------===//

int negative_left_operand_literal(void) {
  return -2 << 2;
  // expected-warning@-1 {{Left operand is negative in left shift}}
  // expected-note@-2 {{The result of left shift is undefined because the left operand is negative}}
}

int negative_left_operand_symbolic(int left, int right) {
  // expected-note@+2 {{Assuming 'left' is < 0}}
  // expected-note@+1 {{Taking false branch}}
  if (left >= 0)
    return 0;
  return left >> right;
  // expected-warning@-1 {{Left operand is negative in right shift}}
  // expected-note@-2 {{The result of right shift is undefined because the left operand is negative}}
}

int negative_left_operand_compound(short arg) {
  // expected-note@+2 {{Assuming 'arg' is < 0}}
  // expected-note@+1 {{Taking false branch}}
  if (arg >= 0)
    return 0;
  return (arg - 3) << 2;
  // expected-warning@-1 {{Left operand is negative in left shift}}
  // expected-note@-2 {{The result of left shift is undefined because the left operand is negative}}
}

int double_negative(void) {
  // In this case we still report that the right operand is negative, because
  // that's the more "serious" issue:
  return -2 >> -2;
  // expected-warning@-1 {{Right operand is negative in right shift}}
  // expected-note@-2 {{The result of right shift is undefined because the right operand is negative}}
}

int single_unknown_negative(int arg) {
  // In this case just one of the operands will be negative, so we end up
  // reporting the left operand after assuming that the right operand is
  // positive.
  // expected-note@+2 {{Assuming 'arg' is not equal to 0}}
  // expected-note@+1 {{Taking false branch}}
  if (!arg)
    return 0;
  // We're first checking the right operand, record that it must be positive,
  // then report that then the left argument must be negative.
  return -arg << arg;
  // expected-warning@-1 {{Left operand is negative in left shift}}
  // expected-note@-2 {{The result of left shift is undefined because the left operand is negative}}
}

void shift_negative_by_zero(int c) {
  // This seems to be innocent, but the standard (before C++20) clearly implies
  // that this is UB, so we should report it in pedantic mode.
  c = (-1) << 0;
  // expected-warning@-1 {{Left operand is negative in left shift}}
  // expected-note@-2 {{The result of left shift is undefined because the left operand is negative}}
}

// TEST OVERFLOW OF CONCRETE SIGNED LEFT OPERAND
//===----------------------------------------------------------------------===//
// (the most complex and least important part of the checker)

int concrete_overflow_literal(void) {
  // 27 in binary is 11011 (5 bits), when shifted by 28 bits it becomes
  // 1_10110000_00000000_00000000_00000000
  return 27 << 28;
  // expected-warning@-1 {{The shift '27 << 28' overflows the capacity of 'int'}}
  // cxx-note@-2 {{The shift '27 << 28' is undefined because 'int' can hold only 32 bits (including the sign bit), so 1 bit overflows}}
  // c-note@-3 {{The shift '27 << 28' is undefined because 'int' can hold only 31 bits (excluding the sign bit), so 2 bits overflow}}
}

int concrete_overflow_symbolic(int arg) {
  // 29 in binary is 11101 (5 bits), when shifted by 29 bits it becomes
  // 11_10100000_00000000_00000000_00000000

  // expected-note@+2 {{Assuming 'arg' is equal to 29}}
  // expected-note@+1 {{Taking false branch}}
  if (arg != 29)
    return 0;
  return arg << arg;
  // expected-warning@-1 {{The shift '29 << 29' overflows the capacity of 'int'}}
  // cxx-note@-2 {{The shift '29 << 29' is undefined because 'int' can hold only 32 bits (including the sign bit), so 2 bits overflow}}
  // c-note@-3 {{The shift '29 << 29' is undefined because 'int' can hold only 31 bits (excluding the sign bit), so 3 bits overflow}}
}

int concrete_overflow_language_difference(void) {
  // 21 in binary is 10101 (5 bits), when shifted by 27 bits it becomes
  // 10101000_00000000_00000000_00000000
  // This does not overflow the 32-bit capacity of int, but reaches the sign
  // bit, which is undefined under C (but accepted in C++ even before C++20).
  return 21 << 27;
  // c-warning@-1 {{The shift '21 << 27' overflows the capacity of 'int'}}
  // c-note@-2 {{The shift '21 << 27' is undefined because 'int' can hold only 31 bits (excluding the sign bit), so 1 bit overflows}}
}

int concrete_overflow_int_min(void) {
  // Another case that's undefined in C but valid in all C++ versions.
  // Note the "represented by 1 bit" special case
  return 1 << 31;
  // c-warning@-1 {{The shift '1 << 31' overflows the capacity of 'int'}}
  // c-note@-2 {{The shift '1 << 31' is undefined because 'int' can hold only 31 bits (excluding the sign bit), so 1 bit overflows}}
}

int concrete_overflow_vague(int arg) {
  // expected-note@+2 {{Assuming 'arg' is > 25}}
  // expected-note@+1 {{Taking false branch}}
  if (arg <= 25)
    return 0;
  return 1024 << arg;
    // expected-warning@-1 {{Left shift of '1024' overflows the capacity of 'int'}}
  // cxx-note@-2 {{Left shift of '1024' is undefined because 'int' can hold only 32 bits (including the sign bit), so some bits overflow}}
  // c-note@-3 {{Left shift of '1024' is undefined because 'int' can hold only 31 bits (excluding the sign bit), so some bits overflow}}
}

int concrete_overflow_vague_only_c(int arg) {
  // A third case that's undefined in C but valid in all C++ versions.

  // c-note@+2 {{Assuming 'arg' is > 20}}
  // c-note@+1 {{Taking false branch}}
  if (arg <= 20)
    return 0;
  return 1024 << arg;
  // c-warning@-1 {{Left shift of '1024' overflows the capacity of 'int'}}
  // c-note@-2 {{Left shift of '1024' is undefined because 'int' can hold only 31 bits (excluding the sign bit), so some bits overflow}}
}

int concrete_overflow_vague_left(int arg) {
  // This kind of overflow check only handles concrete values on the LHS. With
  // some effort it would be possible to report errors in cases like this; but
  // it's probably a waste of time especially considering that overflows of
  // left shifts became well-defined in C++20.

  if (arg <= 1024)
    return 0;
  return arg << 25; // no-warning
}

int concrete_overflow_shift_zero(void) {
  // This is legal, even in C.
  // The relevant rule (as paraphrased on cppreference.com) is:
  // "For signed LHS with nonnegative values, the value of LHS << RHS is
  // LHS * 2^RHS if it is representable in the promoted type of lhs, otherwise
  // the behavior is undefined."
  return 0 << 31; // no-warning
}
