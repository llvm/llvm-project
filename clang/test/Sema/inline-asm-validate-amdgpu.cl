// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -x cl -triple amdgcn -fsyntax-only  %s
// expected-no-diagnostics

kernel void test () {

  int sgpr = 0, vgpr = 0, imm = 0;

  // sgpr constraints
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "s" (imm) : );

  // vgpr constraints
  __asm__ ("v_mov_b32 %0, %1" : "=v" (vgpr) : "v" (imm) : );
}
