// RUN: %clang_cc1 %s -verify

// expected-no-diagnostics

namespace PromotionVersusMutation {
  typedef unsigned Unsigned;
  typedef signed Signed;

  struct T { unsigned n : 2; } t;

  typedef __typeof__(t.n) Unsigned; // Bitfield is unsigned
  typedef __typeof__(+t.n) Signed;  // ... but promotes to signed.

  typedef __typeof__(t.n + 0) Signed; // Arithmetic promotes.

  typedef __typeof__(t.n = 0) Unsigned;  // Assignment produces an lvalue...
  typedef __typeof__(t.n += 0) Unsigned;
  typedef __typeof__(t.n *= 0) Unsigned;
  typedef __typeof__(+(t.n = 0)) Signed;  // ... which is a bit-field.
  typedef __typeof__(+(t.n += 0)) Signed;
  typedef __typeof__(+(t.n *= 0)) Signed;

  typedef __typeof__(++t.n) Unsigned; // Increment is equivalent to compound-assignment.
  typedef __typeof__(--t.n) Unsigned;
  typedef __typeof__(+(++t.n)) Signed;
  typedef __typeof__(+(--t.n)) Signed;

  typedef __typeof__(t.n++) Unsigned; // Post-increment's result has the type
  typedef __typeof__(t.n--) Unsigned; // of the operand...
  typedef __typeof__(+(t.n++)) Unsigned; // ... and is not a bit-field (because
  typedef __typeof__(+(t.n--)) Unsigned; // it's not a glvalue).
}
