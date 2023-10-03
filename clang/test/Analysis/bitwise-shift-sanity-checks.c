// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:    -analyzer-config core.BitwiseShift:Pedantic=true \
// RUN:    -verify=expected,pedantic \
// RUN:    -triple x86_64-pc-linux-gnu -x c %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:    -analyzer-config core.BitwiseShift:Pedantic=true \
// RUN:    -verify=expected,pedantic \
// RUN:    -triple x86_64-pc-linux-gnu -x c++ -std=c++14 %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:    -verify=expected \
// RUN:    -triple x86_64-pc-linux-gnu -x c++ -std=c++20 %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow

// This test file verifies that the BitwiseShift checker does not crash or
// report false positives (at least on the cases that are listed here...)
// Other core checkers are also enabled to see interactions with e.g.
// core.UndefinedBinaryOperatorResult.
// For the sake of brevity, 'note' output is not checked in this file.

// TEST OBVIOUSLY CORRECT CODE
//===----------------------------------------------------------------------===//

unsigned shift_unsigned(void) {
  // Shifts of unsigned LHS may overflow, even if the RHS is signed.
  // In shifts the type of the right operand does not affect the type of the
  // calculation and the result.
  return 1024u << 25ll; // no-warning
}

int shift_zeroes(void) {
  return 0 << 0; // no-warning
}

int no_info(int left, int right) {
  return left << right; // no-warning
}

int all_okay(int left, int right) {
  if (left < 0 || right < 0)
    return 42;
  return (left << right) + (left >> right); // no-warning
}

// DOCUMENT A LIMITATION OF THE ANALYZER ENGINE
//===----------------------------------------------------------------------===//

int signed_arithmetic_good(int left, int right) {
  if (right >= 32)
    return 0;
  return left << (right - 32);
  // expected-warning@-1 {{Right operand is negative in left shift}}
}

int signed_arithmetic_bad(int left, int right) {
  // FIXME: The analyzer engine handles overflow of signed values as if it was
  // a valid code path, so in this case it will think that that (right + 32) is
  // either at least 32 *or* very negative after an overflow.
  // As checkOvershift() is called before checkOperandNegative(), the checker
  // will first rule out the case when (right + 32) is larger than 32 and then
  // it reports that it's negative. Swapping the order of the two checks would
  // trigger an analogous fault in signed_aritmetic_good().
  if (right < 0)
    return 0;
  return left << (right + 32);
  // expected-warning@-1 {{Right operand is negative in left shift}}
  // FIXME: we should rather have {{Left shift overflows the capacity of 'int'}}
}

// TEST THE EXAMPLES FROM THE DOCUMENTATION
//===----------------------------------------------------------------------===//

void basic_examples(int a, int b) {
  if (b < 0) {
    b = a << b; // expected-warning {{Right operand is negative in left shift}}
  } else if (b >= 32) {
    b = a >> b; // expected-warning {{Right shift overflows the capacity of 'int'}}
  }
}

int pedantic_examples(int a, int b) {
  if (a < 0) {
    return a >> b; // pedantic-warning {{Left operand is negative in right shift}}
  }
  a = 1000u << 31; // OK, overflow of unsigned shift is well-defined, a == 0
  if (b > 10) {
    a = b << 31; // this is UB before C++20, but the checker doesn't warn because
                 // it doesn't know the exact value of b
  }
  return 1000 << 31; // pedantic-warning {{The shift '1000 << 31' overflows the capacity of 'int'}}
}

// TEST UNUSUAL CODE THAT SHOULD NOT CRASH
//===----------------------------------------------------------------------===//

__int128 large_left(void) {
  // Ensure that we do not crash when the left operand doesn't fit in 64 bits.
  return (__int128) 1 << 63 << 10 << 10; // no-crash
}

int large_right(void) {
  // Ensure that we do not crash when the right operand doesn't fit in 64 bits.
  return 1 << ((__int128) 1 << 118); // no-crash
  // expected-warning@-1 {{Left shift by '332306998946228968225951765070086144' overflows the capacity of 'int'}}
}

void doubles_cast_to_integer(int *c) {
  *c = 1 << (int)1.5;          // no-crash
  *c = ((int)1.5) << 1;        // no-crash
  *c = ((int)1.5) << (int)1.5; // no-crash
}

// TEST CODE THAT WAS TRIGGERING BUGS IN EARLIER REVISIONS
//===----------------------------------------------------------------------===//

unsigned int strange_cast(unsigned short sh) {
  // This testcase triggers a bug in the constant folding (it "forgets" the
  // cast), which is silenced in SimpleSValBuilder::evalBinOpNN() with an ugly
  // workaround, because otherwise it would lead to a false positive from
  // core.UndefinedBinaryOperatorResult.
  unsigned int i;
  sh++;
  for (i=0; i<sh; i++) {}
  return (unsigned int) ( ((unsigned int) sh) << 16 ); // no-warning
}
