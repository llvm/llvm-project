// RUN: %clang_cc1 -std=c2x -fsyntax-only -verify -Wno-unused %s

// Test that the preprocessor behavior makes sense.
#if 1wb != 1
#error "wb suffix must be recognized by preprocessor"
#endif
#if 1uwb != 1
#error "uwb suffix must be recognized by preprocessor"
#endif
#if !(-1wb < 0)
#error "wb suffix must be interpreted as signed"
#endif
#if !(-1uwb > 0)
#error "uwb suffix must be interpreted as unsigned"
#endif

#if 18446744073709551615uwb != 18446744073709551615ULL
#error "expected the max value for uintmax_t to compare equal"
#endif

// Test that the preprocessor gives appropriate diagnostics when the
// literal value is larger than what can be stored in a [u]intmax_t.
#if 18446744073709551616wb != 0ULL // expected-error {{integer literal is too large to be represented in any integer type}}
#error "never expected to get here due to error"
#endif
#if 18446744073709551616uwb != 0ULL // expected-error {{integer literal is too large to be represented in any integer type}}
#error "never expected to get here due to error"
#endif

// Despite using a bit-precise integer, this is expected to overflow
// because all preprocessor arithmetic is done in [u]intmax_t, so this
// should result in the value 0.
#if 18446744073709551615uwb + 1 != 0ULL
#error "expected modulo arithmetic with uintmax_t width"
#endif

// Because this bit-precise integer is signed, it will also overflow,
// but Clang handles that by converting to uintmax_t instead of
// intmax_t.
#if 18446744073709551615wb + 1 != 0LL // expected-warning {{integer literal is too large to be represented in a signed integer type, interpreting as unsigned}}
#error "expected modulo arithmetic with uintmax_t width"
#endif

// Test that just because the preprocessor can't figure out the bit
// width doesn't mean we can't form the constant, it just means we
// can't use the value in a preprocessor conditional.
unsigned _BitInt(65) Val = 18446744073709551616uwb;

void ValidSuffix(void) {
  // Decimal literals.
  1wb;
  1WB;
  -1wb;
  _Static_assert((int)1wb == 1, "not 1?");
  _Static_assert((int)-1wb == -1, "not -1?");

  1uwb;
  1uWB;
  1Uwb;
  1UWB;
  _Static_assert((unsigned int)1uwb == 1u, "not 1?");

  1'2wb;
  1'2uwb;
  _Static_assert((int)1'2wb == 12, "not 12?");
  _Static_assert((unsigned int)1'2uwb == 12u, "not 12?");

  // Hexadecimal literals.
  0x1wb;
  0x1uwb;
  0x0'1'2'3wb;
  0xA'B'c'duwb;
  _Static_assert((int)0x0'1'2'3wb == 0x0123, "not 0x0123");
  _Static_assert((unsigned int)0xA'B'c'duwb == 0xABCDu, "not 0xABCD");

  // Binary literals.
  0b1wb;
  0b1uwb;
  0b1'0'1'0'0'1wb;
  0b0'1'0'1'1'0uwb;
  _Static_assert((int)0b1wb == 1, "not 1?");
  _Static_assert((unsigned int)0b1uwb == 1u, "not 1?");

  // Octal literals.
  01wb;
  01uwb;
  0'6'0wb;
  0'0'1uwb;
  0wbu;
  0WBu;
  0wbU;
  0WBU;
  0wb;
  _Static_assert((int)0wb == 0, "not 0?");
  _Static_assert((unsigned int)0wbu == 0u, "not 0?");

  // Imaginary or Complex. These are allowed because _Complex can work with any
  // integer type, and that includes _BitInt.
  1iwb;
  1wbj;
}

void InvalidSuffix(void) {
  // Can't mix the case of wb or WB, and can't rearrange the letters.
  0wB; // expected-error {{invalid suffix 'wB' on integer constant}}
  0Wb; // expected-error {{invalid suffix 'Wb' on integer constant}}
  0bw; // expected-error {{invalid digit 'b' in octal constant}}
  0BW; // expected-error {{invalid digit 'B' in octal constant}}

  // Trailing digit separators should still diagnose.
  1'2'wb; // expected-error {{digit separator cannot appear at end of digit sequence}}
  1'2'uwb; // expected-error {{digit separator cannot appear at end of digit sequence}}

  // Long.
  1lwb; // expected-error {{invalid suffix}}
  1wbl; // expected-error {{invalid suffix}}
  1luwb; // expected-error {{invalid suffix}}
  1ulwb;  // expected-error {{invalid suffix}}

  // Long long.
  1llwb; // expected-error {{invalid suffix}}
  1uwbll; // expected-error {{invalid suffix}}

  // Floating point.
  0.1wb;   // expected-error {{invalid suffix}}
  0.1fwb;   // expected-error {{invalid suffix}}

  // Repetitive suffix.
  1wbwb; // expected-error {{invalid suffix}}
  1uwbuwb; // expected-error {{invalid suffix}}
  1wbuwb; // expected-error {{invalid suffix}}
  1uwbwb; // expected-error {{invalid suffix}}
}

void ValidSuffixInvalidValue(void) {
  // This is a valid suffix, but the value is larger than one that fits within
  // the width of BITINT_MAXWIDTH. When this value changes in the future, the
  // test cases should pick a new value that can't be represented by a _BitInt,
  // but also add a test case that a 129-bit literal still behaves as-expected.
  _Static_assert(__BITINT_MAXWIDTH__ <= 128,
	             "Need to pick a bigger constant for the test case below.");
  0xFFFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF'1wb; // expected-error {{integer literal is too large to be represented in any signed integer type}}
  0xFFFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF'FFFF'1uwb; // expected-error {{integer literal is too large to be represented in any integer type}}
}

void TestTypes(void) {
  // 2 value bits, one sign bit
  _Static_assert(__builtin_types_compatible_p(__typeof__(3wb), _BitInt(3)));
  // 2 value bits, one sign bit
  _Static_assert(__builtin_types_compatible_p(__typeof__(-3wb), _BitInt(3)));
  // 2 value bits, no sign bit
  _Static_assert(__builtin_types_compatible_p(__typeof__(3uwb), unsigned _BitInt(2)));
  // 4 value bits, one sign bit
  _Static_assert(__builtin_types_compatible_p(__typeof__(0xFwb), _BitInt(5)));
  // 4 value bits, one sign bit
  _Static_assert(__builtin_types_compatible_p(__typeof__(-0xFwb), _BitInt(5)));
  // 4 value bits, no sign bit
  _Static_assert(__builtin_types_compatible_p(__typeof__(0xFuwb), unsigned _BitInt(4)));
}
