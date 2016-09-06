// REQUIRES: x86-registered-target
// RUN: not %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - 2>&1 | FileCheck %s

void call_x86_64_builtins(void)
{
  unsigned long long a = __builtin_ia32_crc32di(0, 0);
  unsigned long long b;
  unsigned int c = __builtin_ia32_rdseed64_step (&b);
  unsigned long long d = __builtin_ia32_bextr_u64 (0, 0);
  unsigned long long e = __builtin_ia32_pdep_di(0, 0);
  unsigned long long f = __builtin_ia32_pext_di(0, 0);
  unsigned long long g = __builtin_ia32_bzhi_di(0, 0);
  unsigned long long h;
  unsigned long long i = __builtin_ia32_addcarryx_u64(0, 0, 0, &h);
  unsigned long long j;
  unsigned long long k = __builtin_ia32_addcarry_u64(0, 0, 0, &j);
  unsigned long long l;
  unsigned long long m = __builtin_ia32_subborrow_u64(0, 0, 0, &l);
}

// CHECK: error: this builtin is only available on x86-64 targets
// CHECK: __builtin_ia32_crc32di
