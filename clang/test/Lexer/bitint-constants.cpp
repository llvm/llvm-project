// RUN: %clang_cc1 -triple aarch64-unknown-unknown -fsyntax-only -verify -Wno-unused %s

// Test that the preprocessor behavior makes sense.
#if 1__wb != 1
#error "wb suffix must be recognized by preprocessor"
#endif
#if 1__uwb != 1
#error "uwb suffix must be recognized by preprocessor"
#endif
#if !(-1__wb < 0)
#error "wb suffix must be interpreted as signed"
#endif
#if !(-1__uwb > 0)
#error "uwb suffix must be interpreted as unsigned"
#endif

#if 18446744073709551615__uwb != 18446744073709551615ULL
#error "expected the max value for uintmax_t to compare equal"
#endif

// Test that the preprocessor gives appropriate diagnostics when the
// literal value is larger than what can be stored in a [u]intmax_t.
#if 18446744073709551616__wb != 0ULL // expected-error {{integer literal is too large to be represented in any integer type}}
#error "never expected to get here due to error"
#endif
#if 18446744073709551616__uwb != 0ULL // expected-error {{integer literal is too large to be represented in any integer type}}
#error "never expected to get here due to error"
#endif

// Despite using a bit-precise integer, this is expected to overflow
// because all preprocessor arithmetic is done in [u]intmax_t, so this
// should result in the value 0.
#if 18446744073709551615__uwb + 1 != 0ULL
#error "expected modulo arithmetic with uintmax_t width"
#endif

// Because this bit-precise integer is signed, it will also overflow,
// but Clang handles that by converting to uintmax_t instead of
// intmax_t.
#if 18446744073709551615__wb + 1 != 0LL // expected-warning {{integer literal is too large to be represented in a signed integer type, interpreting as unsigned}}
#error "expected modulo arithmetic with uintmax_t width"
#endif

// Test that just because the preprocessor can't figure out the bit
// width doesn't mean we can't form the constant, it just means we
// can't use the value in a preprocessor conditional.
unsigned _BitInt(65) Val = 18446744073709551616__uwb;
// UDL test to make sure underscore parsing is correct
unsigned operator ""_(const char *);

void ValidSuffix(void) {
  // Decimal literals.
  1__wb;
  1__WB;
  -1__wb;
  _Static_assert((int)1__wb == 1, "not 1?");
  _Static_assert((int)-1__wb == -1, "not -1?");

  1__uwb;
  1__uWB;
  1__Uwb;
  1__UWB;
  1u__wb;
  1__WBu;
  1U__WB;
  _Static_assert((unsigned int)1__uwb == 1u, "not 1?");

  1'2__wb;
  1'2__uwb;
  _Static_assert((int)1'2__wb == 12, "not 12?");
  _Static_assert((unsigned int)1'2__uwb == 12u, "not 12?");

  // Hexadecimal literals.
  0x1__wb;
  0x1__uwb;
  0x0'1'2'3__wb;
  0xA'B'c'd__uwb;
  _Static_assert((int)0x0'1'2'3__wb == 0x0123, "not 0x0123");
  _Static_assert((unsigned int)0xA'B'c'd__uwb == 0xABCDu, "not 0xABCD");

  // Binary literals.
  0b1__wb;
  0b1__uwb;
  0b1'0'1'0'0'1__wb;
  0b0'1'0'1'1'0__uwb;
  _Static_assert((int)0b1__wb == 1, "not 1?");
  _Static_assert((unsigned int)0b1__uwb == 1u, "not 1?");

  // Octal literals.
  01__wb;
  01__uwb;
  0'6'0__wb;
  0'0'1__uwb;
  0__wbu;
  0__WBu;
  0U__wb;
  0U__WB;
  0__wb;
  _Static_assert((int)0__wb == 0, "not 0?");
  _Static_assert((unsigned int)0__wbu == 0u, "not 0?");

  // Imaginary or Complex. These are allowed because _Complex can work with any
  // integer type, and that includes _BitInt.
  1__wbi;
  1i__wb;
  1__wbj;

  //UDL test as single underscore
  unsigned i = 1.0_;
}

void InvalidSuffix(void) {
  // Can't mix the case of wb or WB, and can't rearrange the letters.
  0__wB; // expected-error {{invalid suffix '__wB' on integer constant}}
  0__Wb; // expected-error {{invalid suffix '__Wb' on integer constant}}
  0__bw; // expected-error {{invalid suffix '__bw' on integer constant}}
  0__BW; // expected-error {{invalid suffix '__BW' on integer constant}}

  // Trailing digit separators should still diagnose.
  1'2'__wb; // expected-error {{digit separator cannot appear at end of digit sequence}}
  1'2'__uwb; // expected-error {{digit separator cannot appear at end of digit sequence}}

  // Long.
  1l__wb; // expected-error {{invalid suffix}}
  1__wbl; // expected-error {{invalid suffix}}
  1l__uwb; // expected-error {{invalid suffix}}
  1__l; // expected-error {{invalid suffix}}
  1ul__wb;  // expected-error {{invalid suffix}}

  // Long long.
  1ll__wb; // expected-error {{invalid suffix}}
  1__uwbll; // expected-error {{invalid suffix}}

  // Floating point.
  0.1__wb;   // expected-error {{invalid suffix}}
  0.1f__wb;   // expected-error {{invalid suffix}}

  // Repetitive suffix.
  1__wb__wb; // expected-error {{invalid suffix}}
  1__uwbuwb; // expected-error {{invalid suffix}}
  1__wbuwb; // expected-error {{invalid suffix}}
  1__uwbwb; // expected-error {{invalid suffix}}

  // Missing or extra characters in suffix.
  1__; // expected-error {{invalid suffix}}
  1__u; // expected-error {{invalid suffix}}
  1___; // expected-error {{invalid suffix}}
  1___WB; // expected-error {{invalid suffix}}
  1__wb__; // expected-error {{invalid suffix}}
  1__w; // expected-error {{invalid suffix}}
  1__b; // expected-error {{invalid suffix}}
}

void ValidSuffixInvalidValue(void) {
  // This is a valid suffix, but the value is larger than one that fits within
  // the width of BITINT_MAXWIDTH. When this value changes in the future, the
  // test cases should pick a new value that can't be represented by a _BitInt,
  // but also add a test case that a 129-bit literal still behaves as-expected.
  _Static_assert(__BITINT_MAXWIDTH__ <= 128,
	             "Need to pick a bigger constant for the test case below.");
  0xFFFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF'1__wb; // expected-error {{integer literal is too large to be represented in any signed integer type}}
  0xFFFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF'1__uwb; // expected-error {{integer literal is too large to be represented in any integer type}}
}

void TestTypes(void) {
  // 2 value bits, one sign bit
  _Static_assert(__is_same(decltype(3__wb), _BitInt(3)));
  // 2 value bits, one sign bit
  _Static_assert(__is_same(decltype(-3__wb), _BitInt(3)));
  // 2 value bits, no sign bit
  _Static_assert(__is_same(decltype(3__uwb), unsigned _BitInt(2)));
  // 4 value bits, one sign bit
  _Static_assert(__is_same(decltype(0xF__wb), _BitInt(5)));
  // 4 value bits, one sign bit
  _Static_assert(__is_same(decltype(-0xF__wb), _BitInt(5)));
  // 4 value bits, no sign bit
  _Static_assert(__is_same(decltype(0xF__uwb), unsigned _BitInt(4)));
}
