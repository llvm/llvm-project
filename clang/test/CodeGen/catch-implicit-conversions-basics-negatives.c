// RUN: %clang_cc1 -fsanitize=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-recover=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK

// If we have an enum, it will be promoted to an unsigned integer.
// But both types are unsigned, and have same bitwidth.
// So we should not emit any sanitization. Also, for inc/dec we currently assume
// (assert) that we will only have cases where at least one of the types
// is signed, which isn't the case here.
typedef enum { a } b;
b t0(b c) {
  c--;
  return c;
}
