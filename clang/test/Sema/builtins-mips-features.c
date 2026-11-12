// REQUIRES: mips-registered-target
// RUN: %clang_cc1 -triple mips64 -fsyntax-only -verify %s

typedef signed char v4i8 __attribute__ ((vector_size(4)));
typedef signed char v4q7 __attribute__ ((vector_size(4)));
typedef signed char v16i8 __attribute__((vector_size(16), aligned(16)));
typedef unsigned char v16u8 __attribute__((vector_size(16), aligned(16)));

void dsp() {
  v4i8 a;
  void* p;

  // expected-error@+1 {{this builtin requires 'dsp' ASE, please use -mdsp}}
  __builtin_mips_addu_qb(a, a);
  // expected-error@+1 {{this builtin requires 'dsp' ASE, please use -mdsp}}
  __builtin_mips_lwx(p, 32);
}

void dspr2() {
  v4i8 a;
  v4q7 b;

  // expected-error@+1 {{this builtin requires 'dsp r2' ASE, please use -mdspr2}}
  __builtin_mips_absq_s_qb(b);
  // expected-error@+1 {{this builtin requires 'dsp r2' ASE, please use -mdspr2}}
  __builtin_mips_subuh_r_qb(a, a);
}

void msa() {
  v16i8 a;
  v16u8 b;

  // expected-error@+1 {{this builtin requires 'msa' ASE, please use -mmsa}}
  __builtin_msa_add_a_b(a, a);
  // expected-error@+1 {{this builtin requires 'msa' ASE, please use -mmsa}}
  __builtin_msa_xori_b(b, 5);
}
