// Test how BOLT handles indirect branch sequence of instructions in
// AArch64MCPlus builder.
// This test checks that case when we have no shift amount after add
// instruction. This pattern comes from libc, so needs to build '-static'
// binary to reproduce the issue easily.
//
//   adr     x6, 0x219fb0 <sigall_set+0x88>
//   add     x6, x6, x14, lsl #2
//   ldr     w7, [x6]
//   add     x6, x6, w7, sxtw => no shift amount
//   br      x6
// It also tests another case when we use '-fuse-ld=lld' along with '-static'
// which produces the following sequence of intsructions:
//
//  nop   => nop/adr instead of adrp/add
//  adr     x13, 0x215a18 <_nl_value_type_LC_COLLATE+0x50>
//  ldrh    w13, [x13, w12, uxtw #1]
//  adr     x12, 0x247b30 <__gettextparse+0x5b0>
//  add     x13, x12, w13, sxth #2
//  br      x13

// clang-format off

// REQUIRES: system-linux,target=aarch64{{.*}}
// RUN: %clang %s -o %t.exe -Wl,-q -static -fuse-ld=lld \
// RUN: --target=aarch64-unknown-linux-gnu
// RUN: llvm-bolt %t.exe -o %t.bolt --print-cfg \
// RUN:  -v=1 2>&1 | FileCheck --match-full-lines %s

// CHECK: BOLT-WARNING: Failed to match indirect branch: nop/adr instead of adrp/add
// CHECK: BOLT-WARNING: Failed to match indirect branch: ShiftVAL != 2

int main() { return 42; }
