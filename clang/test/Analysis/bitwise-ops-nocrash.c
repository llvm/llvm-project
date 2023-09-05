// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-disable-checker=core.BitwiseShift -analyzer-output=text -triple x86_64-linux-gnu -Wno-shift-count-overflow -verify %s

// NOTE: This test file disables the checker core.BitwiseShift (which would
// report all undefined behavior connected to bitwise shifts) to verify the
// behavior of core.UndefinedBinaryOperatorResult (which resports cases when
// the constant folding in BasicValueFactory produces an "undefined" result
// from a shift or any other binary operator).

#define offsetof(type,memb) ((unsigned long)&((type*)0)->memb)

typedef struct {
  unsigned long guest_counter;
  unsigned int guest_fpc;
} S;

// no crash
int left_shift_overflow_no_crash(unsigned int i) {
  unsigned shift = 323U; // expected-note{{'shift' initialized to 323}}
  switch (i) { // expected-note{{Control jumps to 'case 8:'  at line 20}}
  case offsetof(S, guest_fpc):
    return 3 << shift; // expected-warning{{The result of the left shift is undefined due to shifting by '323', which is greater or equal to the width of type 'int'}}
    // expected-note@-1{{The result of the left shift is undefined due to shifting by '323', which is greater or equal to the width of type 'int'}}
  default:
    break;
  }

  return 0;
}
