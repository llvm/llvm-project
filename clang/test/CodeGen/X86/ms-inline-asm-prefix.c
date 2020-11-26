// REQUIRES: x86-registered-target
// RUN:%clang_cc1 %s -ferror-limit 0 -triple=x86_64-pc-windows-msvc -target-feature +avx512f -target-feature +avx2 -target-feature +avx512vl -fasm-blocks -mllvm -x86-asm-syntax=intel -S -emit-llvm -o -  | FileCheck %s -check-prefix=INTEL
// RUN:%clang_cc1 %s -ferror-limit 0 -triple=x86_64-pc-windows-msvc -target-feature +avx512f -target-feature +avx2 -target-feature +avx512vl -fasm-blocks -mllvm -x86-asm-syntax=att -S -emit-llvm -o -  | FileCheck %s -check-prefix=ATT

void check_inline_prefix(void) {
  __asm {
    // INTEL: call void asm sideeffect inteldialect "{vex} vcvtps2pd xmm0, xmm1\0A\09{vex2} vcvtps2pd xmm0, xmm1\0A\09{vex3} vcvtps2pd xmm0, xmm1\0A\09{evex} vcvtps2pd xmm0, xmm1", "~{xmm0},~{dirflag},~{fpsr},~{flags}"()
    // ATT: call void asm sideeffect inteldialect "{vex} vcvtps2pd xmm0, xmm1\0A\09{vex2} vcvtps2pd xmm0, xmm1\0A\09{vex3} vcvtps2pd xmm0, xmm1\0A\09{evex} vcvtps2pd xmm0, xmm1", "~{xmm0},~{dirflag},~{fpsr},~{flags}"()
    vex vcvtps2pd xmm0, xmm1
    vex2 vcvtps2pd xmm0, xmm1
    vex3 vcvtps2pd xmm0, xmm1
    evex vcvtps2pd xmm0, xmm1
  }
}
