// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -fsanitize=implicit-integer-sign-change -fno-sanitize-recover=implicit-integer-sign-change -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-SANITIZE
// RUN: %clang_cc1 -fsanitize=implicit-integer-sign-change -fsanitize-recover=implicit-integer-sign-change -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-SANITIZE
// RUN: %clang_cc1 -fsanitize=implicit-integer-sign-change -fsanitize-trap=implicit-integer-sign-change -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-SANITIZE

extern "C" { // Disable name mangling.

// ========================================================================== //
// The expected true-negatives.
// ========================================================================== //

// Sanitization is explicitly disabled.
// ========================================================================== //

// CHECK-LABEL: @blacklist_0
__attribute__((no_sanitize("undefined"))) unsigned int blacklist_0(signed int src) {
  // We are not in "undefined" group, so that doesn't work.
  // CHECK-SANITIZE: call
  // CHECK: }
  return src;
}

// CHECK-LABEL: @blacklist_1
__attribute__((no_sanitize("integer"))) unsigned int blacklist_1(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// CHECK-LABEL: @blacklist_2
__attribute__((no_sanitize("implicit-conversion"))) unsigned int blacklist_2(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// CHECK-LABEL: @blacklist_3
__attribute__((no_sanitize("implicit-integer-sign-change"))) unsigned int blacklist_3(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// Explicit sign-changing conversions.
// ========================================================================== //

// CHECK-LABEL: @explicit_signed_int_to_unsigned_int
unsigned int explicit_signed_int_to_unsigned_int(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned int)src;
}

// CHECK-LABEL: @explicit_unsigned_int_to_signed_int
signed int explicit_unsigned_int_to_signed_int(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (signed int)src;
}

// Explicit NOP conversions.
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

// Explicit functional sign-changing casts.
// ========================================================================== //

using UnsignedInt = unsigned int;
using SignedInt = signed int;

// CHECK-LABEL: explicit_functional_unsigned_int_to_signed_int
signed int explicit_functional_unsigned_int_to_signed_int(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return SignedInt(src);
}

// CHECK-LABEL: @explicit_functional_signed_int_to_unsigned_int
unsigned int explicit_functional_signed_int_to_unsigned_int(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return UnsignedInt(src);
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

// Explicit C++-style sign-changing casts.
// ========================================================================== //

// CHECK-LABEL: @explicit_cppstyle_unsigned_int_to_signed_int
signed int explicit_cppstyle_unsigned_int_to_signed_int(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return static_cast<signed int>(src);
}

// CHECK-LABEL: @explicit_cppstyle_signed_int_to_unsigned_int
unsigned int explicit_cppstyle_signed_int_to_unsigned_int(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return static_cast<unsigned int>(src);
}

// Explicit C++-style casts NOP casts.
// ========================================================================== //

// CHECK-LABEL: @explicit_cppstyle_unsigned_int_to_unsigned_int
unsigned int explicit_cppstyle_unsigned_int_to_unsigned_int(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return static_cast<unsigned int>(src);
}

// CHECK-LABEL: @explicit_cppstyle_signed_int_to_signed_int
signed int explicit_cppstyle_signed_int_to_signed_int(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return static_cast<signed int>(src);
}

} // extern "C"
