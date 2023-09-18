// RUN: %clang_analyze_cc1 -analyzer-checker=core.BitwiseShift \
// RUN:    -analyzer-output=text -verify \
// RUN:    -triple x86_64-pc-linux-gnu -x c %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core.BitwiseShift \
// RUN:    -analyzer-config core.BitwiseShift:Pedantic=true \
// RUN:    -analyzer-output=text -verify \
// RUN:    -triple x86_64-pc-linux-gnu -x c++ -std=c++20 %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow
//
// This test file verifies the default behavior of the BitwiseShift checker,
// which reports the serious logical defects, but doesn't warn on code that's
// legal under C++20 (or later) and widely accepted (but theoretically
// undefined) in other compilation modes.

// TEST NEGATIVE RIGHT OPERAND
//===----------------------------------------------------------------------===//

int negative_right_operand_literal(void) {
  return 2 << -2;
  // expected-warning@-1 {{Right operand is negative in left shift}}
  // expected-note@-2 {{The result of left shift is undefined because the right operand is negative}}
}

int negative_right_operand_symbolic(int left, int right) {
  // expected-note@+2 {{Assuming 'right' is < 0}}
  // expected-note@+1 {{Taking false branch}}
  if (right >= 0)
    return 0;
  return left >> right;
  // expected-warning@-1 {{Right operand is negative in right shift}}
  // expected-note@-2 {{The result of right shift is undefined because the right operand is negative}}
}

int negative_right_operand_compound(short arg) {
  // expected-note@+2 {{Assuming 'arg' is < 2}}
  // expected-note@+1 {{Taking false branch}}
  if (arg >= 2 )
    return 0;
  return 2 << (arg - 1 - 1 - 1);
  // expected-warning@-1 {{Right operand is negative in left shift}}
  // expected-note@-2 {{The result of left shift is undefined because the right operand is negative}}
}

// TEST TOO LARGE RIGHT OPERAND
//===----------------------------------------------------------------------===//

int too_large_right_operand_literal(void) {
  return 2 << 32;
  // expected-warning@-1 {{Left shift by '32' overflows the capacity of 'int'}}
  // expected-note@-2 {{The result of left shift is undefined because the right operand '32' is not smaller than 32, the capacity of 'int'}}
}

int too_large_right_operand_exact_symbolic(int arg) {
  // expected-note@+4 {{Assuming 'arg' is > 33}}
  // expected-note@+3 {{Left side of '||' is false}}
  // expected-note@+2 {{Assuming 'arg' is < 35}}
  // expected-note@+1 {{Taking false branch}}
  if (arg <= 33 || arg >= 35)
    return 0;
  return 3 << arg;
  // expected-warning@-1 {{Left shift by '34' overflows the capacity of 'int'}}
  // expected-note@-2 {{The result of left shift is undefined because the right operand '34' is not smaller than 32, the capacity of 'int'}}
}

int too_large_right_operand_exact_symbolic_2(char arg) {
  // expected-note@+2 {{Assuming the condition is false}}
  // expected-note@+1 {{Taking false branch}}
  if (arg != ' ')
    return 0;
  return 3 << arg;
  // expected-warning@-1 {{Left shift by '32' overflows the capacity of 'int'}}
  // expected-note@-2 {{The result of left shift is undefined because the right operand '32' is not smaller than 32, the capacity of 'int'}}
}

int too_large_right_operand_symbolic(int left, int right) {
  // expected-note@+2 {{Assuming 'right' is > 31}}
  // expected-note@+1 {{Taking false branch}}
  if (right <= 31)
    return 0;
  return left >> right;
  // expected-warning@-1 {{Right shift overflows the capacity of 'int'}}
  // expected-note@-2 {{The result of right shift is undefined because the right operand is not smaller than 32, the capacity of 'int'}}
  // NOTE: the messages of the checker are a bit vague in this case, but the
  // tracking of the variables reveals our knowledge about them.
}

int too_large_right_operand_compound(unsigned short arg) {
  // Note: this would be valid code with an 'unsigned int' because
  // unsigned addition is allowed to overflow.
  return 1 << (32 + arg);
  // expected-warning@-1 {{Left shift overflows the capacity of 'int'}}
  // expected-note@-2 {{The result of left shift is undefined because the right operand is not smaller than 32, the capacity of 'int'}}
}

// TEST STATE UPDATES
//===----------------------------------------------------------------------===//

void state_update(char a, int *p) {
  // NOTE: with 'int a' this would not produce a bug report because the engine
  // would not rule out an overflow.
  *p += 1 << a;
  // expected-note@-1 {{Assuming right operand of bit shift is non-negative but less than 32}}
  *p += 1 << (a + 32);
  // expected-warning@-1 {{Left shift overflows the capacity of 'int'}}
  // expected-note@-2 {{The result of left shift is undefined because the right operand is not smaller than 32, the capacity of 'int'}}
}

void state_update_2(char a, int *p) {
  *p += 1234 >> (a + 32);
  // expected-note@-1 {{Assuming right operand of bit shift is non-negative but less than 32}}
  *p += 1234 >> a;
  // expected-warning@-1 {{Right operand is negative in right shift}}
  // expected-note@-2 {{The result of right shift is undefined because the right operand is negative}}
}

// TEST EXPRESSION TRACKING
//===----------------------------------------------------------------------===//
// Expression tracking a "generic" tool that's used by many other checkers,
// so this is just a minimal test to see that it's activated.

void setValue(unsigned *p, unsigned newval) {
  *p = newval;
  // expected-note@-1 {{The value 33 is assigned to 'right'}}
}

int expression_tracked_back(void) {
  unsigned left = 115; // expected-note {{'left' initialized to 115}}
  unsigned right;
  setValue(&right, 33);
  // expected-note@-1 {{Calling 'setValue'}}
  // expected-note@-2 {{Passing the value 33 via 2nd parameter 'newval'}}
  // expected-note@-3 {{Returning from 'setValue'}}

  return left << right;
  // expected-warning@-1 {{Left shift by '33' overflows the capacity of 'unsigned int'}}
  // expected-note@-2 {{The result of left shift is undefined because the right operand '33' is not smaller than 32, the capacity of 'unsigned int'}}
}

// TEST PERMISSIVENESS
//===----------------------------------------------------------------------===//

int allow_overflows_and_negative_operands(void) {
  // These are all legal under C++ 20 and many compilers accept them under
  // earlier standards as well.
  int int_min = 1 << 31; // no-warning
  int this_overflows = 1027 << 30; // no-warning
  return (-2 << 5) + (-3 >> 4); // no-warning
}

int double_negative(void) {
  return -2 >> -2;
  // expected-warning@-1 {{Right operand is negative in right shift}}
  // expected-note@-2 {{The result of right shift is undefined because the right operand is negative}}
}
