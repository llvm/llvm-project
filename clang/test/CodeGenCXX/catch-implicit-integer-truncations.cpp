// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER
// RUN: %clang_cc1 -fsanitize=implicit-integer-truncation -fsanitize-recover=implicit-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -fsanitize=implicit-integer-truncation -fsanitize-trap=implicit-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP

extern "C" { // Disable name mangling.

// ========================================================================== //
// Check that explicit cast does not interfere with implicit conversion
// ========================================================================== //
// These contain one implicit truncating conversion, and one explicit truncating cast.
// We want to make sure that we still diagnose the implicit conversion.

// Implicit truncation after explicit truncation.
// CHECK-LABEL: @explicit_cast_interference0
unsigned char explicit_cast_interference0(unsigned int c) {
  // CHECK-SANITIZE: %[[ANYEXT:.*]] = zext i8 %[[DST:.*]] to i16, !nosanitize
  // CHECK-SANITIZE: call
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned short)c;
}

// Implicit truncation before explicit truncation.
// CHECK-LABEL: @explicit_cast_interference1
unsigned char explicit_cast_interference1(unsigned int c) {
  // CHECK-SANITIZE: %[[ANYEXT:.*]] = zext i16 %[[DST:.*]] to i32, !nosanitize
  // CHECK-SANITIZE: call
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  unsigned short b;
  return (unsigned char)(b = c);
}

// ========================================================================== //
// The expected true-negatives.
// ========================================================================== //

// Explicit truncating casts.
// ========================================================================== //

// CHECK-LABEL: @explicit_unsigned_int_to_unsigned_char
unsigned char explicit_unsigned_int_to_unsigned_char(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned char)src;
}

// CHECK-LABEL: @explicit_signed_int_to_unsigned_char
unsigned char explicit_signed_int_to_unsigned_char(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned char)src;
}

// CHECK-LABEL: @explicit_unsigned_int_to_signed_char
signed char explicit_unsigned_int_to_signed_char(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (signed char)src;
}

// CHECK-LABEL: @explicit_signed_int_to_signed_char
signed char explicit_signed_int_to_signed_char(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (signed char)src;
}

// Explicit NOP casts.
// ========================================================================== //

// CHECK-LABEL: @explicit_unsigned_int_to_unsigned_int
unsigned int explicit_unsigned_int_to_unsigned_int(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned int)src;
}

// CHECK-LABEL: @explicit_signed_int_to_signed_int
signed int explicit_signed_int_to_signed_int(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (signed int)src;
}

// CHECK-LABEL: @explicit_unsigned_char_to_signed_char
unsigned char explicit_unsigned_char_to_signed_char(unsigned char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned char)src;
}

// CHECK-LABEL: @explicit_signed_char_to_signed_char
signed char explicit_signed_char_to_signed_char(signed char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (signed char)src;
}

// Explicit functional truncating casts.
// ========================================================================== //

using UnsignedChar = unsigned char;
using SignedChar = signed char;
using UnsignedInt = unsigned int;
using SignedInt = signed int;

// CHECK-LABEL: @explicit_functional_unsigned_int_to_unsigned_char
unsigned char explicit_functional_unsigned_int_to_unsigned_char(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return UnsignedChar(src);
}

// CHECK-LABEL: @explicit_functional_signed_int_to_unsigned_char
unsigned char explicit_functional_signed_int_to_unsigned_char(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return UnsignedChar(src);
}

// CHECK-LABEL: @explicit_functional_unsigned_int_to_signed_char
signed char explicit_functional_unsigned_int_to_signed_char(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return SignedChar(src);
}

// CHECK-LABEL: @explicit_functional_signed_int_to_signed_char
signed char explicit_functional_signed_int_to_signed_char(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return SignedChar(src);
}

// Explicit functional NOP casts.
// ========================================================================== //

// CHECK-LABEL: @explicit_functional_unsigned_int_to_unsigned_int
unsigned int explicit_functional_unsigned_int_to_unsigned_int(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return UnsignedInt(src);
}

// CHECK-LABEL: @explicit_functional_signed_int_to_signed_int
signed int explicit_functional_signed_int_to_signed_int(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return SignedInt(src);
}

// CHECK-LABEL: @explicit_functional_unsigned_char_to_signed_char
unsigned char explicit_functional_unsigned_char_to_signed_char(unsigned char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return UnsignedChar(src);
}

// CHECK-LABEL: @explicit_functional_signed_char_to_signed_char
signed char explicit_functional_signed_char_to_signed_char(signed char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return SignedChar(src);
}

// Explicit C++-style casts truncating casts.
// ========================================================================== //

// CHECK-LABEL: @explicit_cppstyleunsigned_int_to_unsigned_char
unsigned char explicit_cppstyleunsigned_int_to_unsigned_char(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return static_cast<unsigned char>(src);
}

// CHECK-LABEL: @explicit_cppstylesigned_int_to_unsigned_char
unsigned char explicit_cppstylesigned_int_to_unsigned_char(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return static_cast<unsigned char>(src);
}

// CHECK-LABEL: @explicit_cppstyleunsigned_int_to_signed_char
signed char explicit_cppstyleunsigned_int_to_signed_char(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return static_cast<signed char>(src);
}

// CHECK-LABEL: @explicit_cppstylesigned_int_to_signed_char
signed char explicit_cppstylesigned_int_to_signed_char(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return static_cast<signed char>(src);
}

// Explicit C++-style casts NOP casts.
// ========================================================================== //

// CHECK-LABEL: @explicit_cppstyleunsigned_int_to_unsigned_int
unsigned int explicit_cppstyleunsigned_int_to_unsigned_int(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return static_cast<unsigned int>(src);
}

// CHECK-LABEL: @explicit_cppstylesigned_int_to_signed_int
signed int explicit_cppstylesigned_int_to_signed_int(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return static_cast<signed int>(src);
}

// CHECK-LABEL: @explicit_cppstyleunsigned_char_to_signed_char
unsigned char explicit_cppstyleunsigned_char_to_signed_char(unsigned char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return static_cast<unsigned char>(src);
}

// CHECK-LABEL: @explicit_cppstylesigned_char_to_signed_char
signed char explicit_cppstylesigned_char_to_signed_char(signed char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return static_cast<signed char>(src);
}

} // extern "C"

// ---------------------------------------------------------------------------//
// A problematic true-negative involving simple C++ code.
// The problem is tha the NoOp ExplicitCast is directly within MaterializeTemporaryExpr(),
// so a special care is neeeded.
// See https://reviews.llvm.org/D48958#1161345
template <typename a>
a b(a c, const a &d) {
  if (d)
    ;
  return c;
}

extern "C" { // Disable name mangling.

// CHECK-LABEL: @false_positive_with_MaterializeTemporaryExpr
int false_positive_with_MaterializeTemporaryExpr() {
  // CHECK-SANITIZE-NOT: call{{.*}}ubsan
  // CHECK: }
  int e = b<unsigned>(4, static_cast<unsigned>(4294967296));
  return e;
}

// ---------------------------------------------------------------------------//

} // extern "C"
