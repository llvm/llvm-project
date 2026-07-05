// REQUIRES: x86-registered-target

// RUN:%clang_cc1 %s -ferror-limit 0 -triple=x86_64-pc -target-feature +avx512f -target-feature +avx2 -target-feature +avx512vl -S -o -  | FileCheck %s -check-prefix CHECK

// This test is to check if the prefix in inline assembly is correctly
// preserved.

void check_inline_prefix(void) {
  __asm__ (
// CHECK: vcvtps2pd %xmm0, %xmm1
// CHECK: {vex} vcvtps2pd %xmm0, %xmm1
// CHECK: {vex2} vcvtps2pd %xmm0, %xmm1
// CHECK: {vex3} vcvtps2pd %xmm0, %xmm1
// CHECK: {evex} vcvtps2pd %xmm0, %xmm1
// CHECK: movl $1, (%rax)
// CHECK: {disp8}  movl $1, (%rax)
// CHECK: {disp32} movl $1, (%rax)
    "vcvtps2pd %xmm0, %xmm1\n\t"
    "{vex} vcvtps2pd %xmm0, %xmm1\n\t"
    "{vex2} vcvtps2pd %xmm0, %xmm1\n\t"
    "{vex3} vcvtps2pd %xmm0, %xmm1\n\t"
    "{evex} vcvtps2pd %xmm0, %xmm1\n\t"
    "movl $1, (%rax)\n\t"
    "{disp8} movl $1, (%rax)\n\t"
    "{disp32} movl $1, (%rax)\n\t"
  );
}
