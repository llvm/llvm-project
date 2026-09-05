// RUN: %clang_cc1 -triple aarch64-linux-gnu -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-IR
// RUN: %clang_cc1 -triple aarch64-linux-gnu -O2 -S -o - %s | FileCheck %s --check-prefix=CHECK-ASM

// Test the "rZ" inline assembly constraint for AArch64.

// CHECK-IR-LABEL: define dso_local void @test_rZ_zero_i64(
// CHECK-IR: tail call void asm sideeffect "str $1, [$0]", "r,^rZ"(ptr %addr, i64 0)
//
// CHECK-ASM-LABEL: test_rZ_zero_i64:
// CHECK-ASM: str xzr, [x0]
void test_rZ_zero_i64(long *addr) {
    __asm__ volatile("str %1, [%0]" : : "r"(addr), "rZ"(0L));
}

// CHECK-IR-LABEL: define dso_local void @test_rZ_zero_i32(
// CHECK-IR: tail call void asm sideeffect "str ${1:w}, [$0]", "r,^rZ"(ptr %addr, i32 0)
//
// CHECK-ASM-LABEL: test_rZ_zero_i32:
// CHECK-ASM: str wzr, [x0]
void test_rZ_zero_i32(int *addr) {
    __asm__ volatile("str %w1, [%0]" : : "r"(addr), "rZ"(0));
}

// CHECK-IR-LABEL: define dso_local void @test_rZ_zero_i16(
// CHECK-IR: tail call void asm sideeffect "strh ${1:w}, [$0]", "r,^rZ"(ptr %addr, i16 0)
//
// CHECK-ASM-LABEL: test_rZ_zero_i16:
// CHECK-ASM: strh wzr, [x0]
void test_rZ_zero_i16(short *addr) {
    __asm__ volatile("strh %w1, [%0]" : : "r"(addr), "rZ"((short)0));
}

// CHECK-IR-LABEL: define dso_local void @test_rZ_zero_i8(
// CHECK-IR: tail call void asm sideeffect "strb ${1:w}, [$0]", "r,^rZ"(ptr %addr, i8 0)
//
// CHECK-ASM-LABEL: test_rZ_zero_i8:
// CHECK-ASM: strb wzr, [x0]
void test_rZ_zero_i8(char *addr) {
    __asm__ volatile("strb %w1, [%0]" : : "r"(addr), "rZ"((char)0));
}

// CHECK-IR-LABEL: define dso_local void @test_rz_lowercase(
// CHECK-IR: tail call void asm sideeffect "str $1, [$0]", "r,^rz"(ptr %addr, i64 0)
//
// CHECK-ASM-LABEL: test_rz_lowercase:
// CHECK-ASM: str xzr, [x0]
void test_rz_lowercase(long *addr) {
    __asm__ volatile("str %1, [%0]" : : "r"(addr), "rz"(0L));
}

// CHECK-IR-LABEL: define dso_local void @test_rZ_explicit_x(
// CHECK-IR: tail call void asm sideeffect "mov ${0:x}, xzr", "^rZ"(i64 0)
//
// CHECK-ASM-LABEL: test_rZ_explicit_x:
// CHECK-ASM: mov xzr, xzr
void test_rZ_explicit_x(void) {
    __asm__ volatile("mov %x0, xzr" : : "rZ"(0L));
}

// CHECK-IR-LABEL: define dso_local void @test_rZ_explicit_w(
// CHECK-IR: tail call void asm sideeffect "mov ${0:w}, wzr", "^rZ"(i32 0)
//
// CHECK-ASM-LABEL: test_rZ_explicit_w:
// CHECK-ASM: mov wzr, wzr
void test_rZ_explicit_w(void) {
    __asm__ volatile("mov %w0, wzr" : : "rZ"(0));
}

// CHECK-IR-LABEL: define dso_local void @test_rZ_x_modifier(
// CHECK-IR: tail call void asm sideeffect "add x2, x1, ${0:x}", "^rZ"(i64 0)
//
// CHECK-ASM-LABEL: test_rZ_x_modifier:
// CHECK-ASM: add x2, x1, xzr
void test_rZ_x_modifier(void) {
    __asm__ volatile("add x2, x1, %x0" : : "rZ"(0L));
}

// CHECK-IR-LABEL: define dso_local void @test_rZ_w_modifier(
// CHECK-IR: tail call void asm sideeffect "add w2, w1, ${0:w}", "^rZ"(i32 0)
//
// CHECK-ASM-LABEL: test_rZ_w_modifier:
// CHECK-ASM: add w2, w1, wzr
void test_rZ_w_modifier(void) {
    __asm__ volatile("add w2, w1, %w0" : : "rZ"(0));
}

