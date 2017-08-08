// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s \
// RUN:     -fsanitize=signed-integer-overflow \
// RUN:   | FileCheck %s

// Verify we emit constants for "immediate" inline assembly arguments.
// Emitting a scalar expression can make the immediate be generated as
// overflow intrinsics, if the overflow sanitizer is enabled.

// Check both 'i' and 'I':
// - 'i' accepts symbolic constants.
// - 'I' doesn't, and is really an immediate-required constraint.

// See also PR23517.

// CHECK-LABEL: @test_inlineasm_i
// CHECK: call void asm sideeffect "int $0", "i{{.*}}"(i32 2)
void test_inlineasm_i() {
  __asm__ __volatile__("int %0" :: "i"(1 + 1));
}

// CHECK-LABEL: @test_inlineasm_I
// CHECK: call void asm sideeffect "int $0", "I{{.*}}"(i32 2)
// CHECK: call void asm sideeffect "int $0", "I{{.*}}"(i32 3)
void test_inlineasm_I() {
  __asm__ __volatile__("int %0" :: "I"(1 + 1));

  // Also check a C non-ICE.
  static const int N = 1;
  __asm__ __volatile__("int %0" :: "I"(N + 2));
}
