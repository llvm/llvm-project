// Verify that ignorelist sections correctly select sanitizers to apply ignorelist entries to.
//
// RUN: %clang_cc1 -fsanitize=unsigned-integer-overflow,cfi-icall -fsanitize-ignorelist=%S/Inputs/sanitizer-special-case-list.unsanitized1.txt -emit-llvm %s -o - | FileCheck %s --check-prefix=UNSANITIZED
// RUN: %clang_cc1 -fsanitize=unsigned-integer-overflow,cfi-icall -fsanitize-ignorelist=%S/Inputs/sanitizer-special-case-list.unsanitized2.txt -emit-llvm %s -o - | FileCheck %s --check-prefix=UNSANITIZED
// RUN: %clang_cc1 -fsanitize=unsigned-integer-overflow,cfi-icall -fsanitize-ignorelist=%S/Inputs/sanitizer-special-case-list.unsanitized3.txt -emit-llvm %s -o - | FileCheck %s --check-prefix=UNSANITIZED
// RUN: %clang_cc1 -fsanitize=unsigned-integer-overflow,cfi-icall -fsanitize-ignorelist=%S/Inputs/sanitizer-special-case-list.unsanitized4.txt -emit-llvm %s -o - | FileCheck %s --check-prefix=UNSANITIZED
//
// RUN: %clang_cc1 -fsanitize=unsigned-integer-overflow,cfi-icall -fsanitize-ignorelist=%S/Inputs/sanitizer-special-case-list.sanitized.txt -emit-llvm %s -o - | FileCheck %s --check-prefix=SANITIZED

unsigned i;

// SANITIZED: @overflow
// UNSANITIZED: @overflow
unsigned overflow(void) {
  // SANITIZED: call {{.*}}void @__ubsan
  // UNSANITIZED-NOT: call {{.*}}void @__ubsan
  return i * 37;
}

// SANITIZED: @cfi
// UNSANITIZED: @cfi
void cfi(void (*fp)(void)) {
  // SANITIZED: llvm.type.test
  // UNSANITIZED-NOT: llvm.type.test
  fp();
}
