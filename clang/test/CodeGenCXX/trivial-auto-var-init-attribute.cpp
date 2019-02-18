// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fblocks %s -emit-llvm -o - | FileCheck %s -check-prefix=UNINIT
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fblocks -ftrivial-auto-var-init=pattern %s -emit-llvm -o - | FileCheck %s -check-prefix=PATTERN
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fblocks -ftrivial-auto-var-init=zero %s -emit-llvm -o - | FileCheck %s -check-prefix=ZERO

template<typename T> void used(T &) noexcept;

extern "C" {

// UNINIT-LABEL:  test_attribute_uninitialized(
// UNINIT:      alloca
// UNINIT-NEXT: call void
// ZERO-LABEL:    test_attribute_uninitialized(
// ZERO:      alloca
// ZERO-NEXT: call void
// PATTERN-LABEL: test_attribute_uninitialized(
// PATTERN:      alloca
// PATTERN-NEXT: call void
void test_attribute_uninitialized() {
  [[clang::uninitialized]] int i;
  used(i);
}

} // extern "C"
