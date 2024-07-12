// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fsyntax-only -verify %s

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void test () {

  int sgpr = 0, vgpr = 0, imm = 0;

  // sgpr constraints
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "s" (imm) : );

  __asm__ ("s_mov_b32 %0, %1" : "={s1}" (sgpr) : "{exec}" (imm) : );
  __asm__ ("s_mov_b32 %0, %1" : "={s1}" (sgpr) : "{exe" (imm) : ); // expected-error {{invalid input constraint '{exe' in asm}}
  __asm__ ("s_mov_b32 %0, %1" : "={s1}" (sgpr) : "{exec" (imm) : ); // expected-error {{invalid input constraint '{exec' in asm}}
  __asm__ ("s_mov_b32 %0, %1" : "={s1}" (sgpr) : "{exec}a" (imm) : ); // expected-error {{invalid input constraint '{exec}a' in asm}}

  // vgpr constraints
  __asm__ ("v_mov_b32 %0, %1" : "=v" (vgpr) : "v" (imm) : );

  // 'I' constraint (an immediate integer in the range -16 to 64)
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "I" (imm) : );
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "I" (-16) : );
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "I" (64) : );
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "I" (-17) : ); // expected-error {{value '-17' out of range for constraint 'I'}}
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "I" (65) : ); // expected-error {{value '65' out of range for constraint 'I'}}

  // 'J' constraint (an immediate 16-bit signed integer)
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "J" (imm) : );
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "J" (-32768) : );
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "J" (32767) : );
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "J" (-32769) : ); // expected-error {{value '-32769' out of range for constraint 'J'}}
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "J" (32768) : ); // expected-error {{value '32768' out of range for constraint 'J'}}

  // 'A' constraint (an immediate constant that can be inlined)
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "A" (imm) : );

  // 'B' constraint (an immediate 32-bit signed integer)
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "B" (imm) : );

  // 'C' constraint (an immediate 32-bit unsigned integer or 'A' constraint)
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "C" (imm) : );

  // 'DA' constraint (an immediate 64-bit constant that can be split into two 'A' constants)
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "DA" (imm) : );

  // 'DB' constraint (an immediate 64-bit constant that can be split into two 'B' constants)
  __asm__ ("s_mov_b32 %0, %1" : "=s" (sgpr) : "DB" (imm) : );

}

__kernel void
test_float(const __global float *a, const __global float *b, __global float *c, unsigned i)
{
    float ai = a[i];
    float bi = b[i];
    float ci;

    __asm("v_add_f32_e32 v1, v2, v3" : "={v1}"(ci) : "{v2}"(ai), "{v3}"(bi) : );
    __asm("v_add_f32_e32 v1, v2, v3" : ""(ci) : "{v2}"(ai), "{v3}"(bi) : ); // expected-error {{invalid output constraint '' in asm}}
    __asm("v_add_f32_e32 v1, v2, v3" : "="(ci) : "{v2}"(ai), "{v3}"(bi) : ); // expected-error {{invalid output constraint '=' in asm}}
    __asm("v_add_f32_e32 v1, v2, v3" : "={a}"(ci) : "{v2}"(ai), "{v3}"(bi) : ); // expected-error {{invalid output constraint '={a}' in asm}}
    __asm("v_add_f32_e32 v1, v2, v3" : "={"(ci) : "{v2}"(ai), "{v3}"(bi) : ); // expected-error {{invalid output constraint '={' in asm}}
    __asm("v_add_f32_e32 v1, v2, v3" : "={}"(ci) : "{v2}"(ai), "{v3}"(bi) : ); // expected-error {{invalid output constraint '={}' in asm}}
    __asm("v_add_f32_e32 v1, v2, v3" : "={v"(ci) : "{v2}"(ai), "{v3}"(bi) : ); // expected-error {{invalid output constraint '={v' in asm}}
    __asm("v_add_f32_e32 v1, v2, v3" : "={v1a}"(ci) : "{v2}"(ai), "{v3}"(bi) : ); // expected-error {{invalid output constraint '={v1a}' in asm}}
    __asm("v_add_f32_e32 v1, v2, v3" : "={va}"(ci) : "{v2}"(ai), "{v3}"(bi) : ); // expected-error {{invalid output constraint '={va}' in asm}}
    __asm("v_add_f32_e32 v1, v2, v3" : "={v1}a"(ci) : "{v2}"(ai), "{v3}"(bi) : ); // expected-error {{invalid output constraint '={v1}a' in asm}}
    __asm("v_add_f32_e32 v1, v2, v3" : "={v1"(ci) : "{v2}"(ai), "{v3}"(bi) : ); // expected-error {{invalid output constraint '={v1' in asm}}
    __asm("v_add_f32_e32 v1, v2, v3" : "=v1}"(ci) : "{v2}"(ai), "{v3}"(bi) : ); // expected-error {{invalid output constraint '=v1}' in asm}}

    __asm("v_add_f32_e32 v1, v2, v3" : "={v[1]}"(ci) : "{v[2]}"(ai), "{v[3]}"(bi) : );
    __asm("v_add_f32_e32 v1, v2, v3" : "={v[1}"(ci) : "{v[2]}"(ai), "{v[3]}"(bi) : ); // expected-error {{invalid output constraint '={v[1}' in asm}}
    __asm("v_add_f32_e32 v1, v2, v3" : "={v[1]"(ci) : "{v[2]}"(ai), "{v[3]}"(bi) : ); // expected-error {{invalid output constraint '={v[1]' in asm}}
    __asm("v_add_f32_e32 v1, v2, v3" : "={v[a]}"(ci) : "{v[2]}"(ai), "{v[3]}"(bi) : ); // expected-error {{invalid output constraint '={v[a]}' in asm}}

    __asm("v_add_f32_e32 v1, v2, v3" : "=v"(ci) : "v"(ai), "v"(bi) : );
    __asm("v_add_f32_e32 v1, v2, v3" : "=v1"(ci) : "v2"(ai), "v3"(bi) : ); /// expected-error {{invalid output constraint '=v1' in asm}}

    __asm("v_add_f32_e32 v1, v2, v3" : "={v1}"(ci) : "{a}"(ai), "{v3}"(bi) : ); // expected-error {{invalid input constraint '{a}' in asm}}
    __asm("v_add_f32_e32 v1, v2, v3" : "={v1}"(ci) : "{v2}"(ai), "{a}"(bi) : ); // expected-error {{invalid input constraint '{a}' in asm}}
    c[i] = ci;
}

__kernel void
test_double(const __global double *a, const __global double *b, __global double *c, unsigned i)
{
    double ai = a[i];
    double bi = b[i];
    double ci;

    __asm("v_add_f64_e64 v[1:2], v[3:4], v[5:6]" : "={v[1:2]}"(ci) : "{v[3:4]}"(ai), "{v[5:6]}"(bi) : );
    __asm("v_add_f64_e64 v[1:2], v[3:4], v[5:6]" : "=v{[1:2]}"(ci) : "{v[3:4]}"(ai), "{v[5:6]}"(bi) : ); //expected-error {{invalid output constraint '=v{[1:2]}' in asm}}
    __asm("v_add_f64_e64 v[1:2], v[3:4], v[5:6]" : "={v[1:2]a}"(ci) : "{v[3:4]}"(ai), "{v[5:6]}"(bi) : ); //expected-error {{invalid output constraint '={v[1:2]a}' in asm}}
    __asm("v_add_f64_e64 v[1:2], v[3:4], v[5:6]" : "={v[1:2]}a"(ci) : "{v[3:4]}"(ai), "{v[5:6]}"(bi) : ); //expected-error {{invalid output constraint '={v[1:2]}a' in asm}}
    __asm("v_add_f64_e64 v[1:2], v[3:4], v[5:6]" : "={v[1:"(ci) : "{v[3:4]}"(ai), "{v[5:6]}"(bi) : ); //expected-error {{invalid output constraint '={v[1:' in asm}}
    __asm("v_add_f64_e64 v[1:2], v[3:4], v[5:6]" : "={v[1:]}"(ci) : "{v[3:4]}"(ai), "{v[5:6]}"(bi) : ); //expected-error {{invalid output constraint '={v[1:]}' in asm}}
    __asm("v_add_f64_e64 v[1:2], v[3:4], v[5:6]" : "={v[:2]}"(ci) : "{v[3:4]}"(ai), "{v[5:6]}"(bi) : ); //expected-error {{invalid output constraint '={v[:2]}' in asm}}
    __asm("v_add_f64_e64 v[1:2], v[3:4], v[5:6]" : "={v[1:2]"(ci) : "{v[3:4]}"(ai), "{v[5:6]}"(bi) : ); //expected-error {{invalid output constraint '={v[1:2]' in asm}}
    __asm("v_add_f64_e64 v[1:2], v[3:4], v[5:6]" : "={v[1:2}"(ci) : "{v[3:4]}"(ai), "{v[5:6]}"(bi) : ); //expected-error {{invalid output constraint '={v[1:2}' in asm}}
    __asm("v_add_f64_e64 v[1:2], v[3:4], v[5:6]" : "={v[2:1]}"(ci) : "{v[3:4]}"(ai), "{v[5:6]}"(bi) : ); //expected-error {{invalid output constraint '={v[2:1]}' in asm}}

    __asm("v_add_f64_e64 v[1:2], v[3:4], v[5:6]" : "=v[1:2]"(ci) : "v[3:4]"(ai), "v[5:6]"(bi) : ); //expected-error {{invalid output constraint '=v[1:2]' in asm}}

    c[i] = ci;
}

void test_long(int arg0) {
  long v15_16;
  __asm volatile("v_lshlrev_b64 v[15:16], 0, %0" : "={v[15:16]}"(v15_16) : "v"(arg0));
}
