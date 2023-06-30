// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -emit-llvm -o - %s
// RUN: %clang_cc1 -verify -triple x86_64-windows-pc -fms-compatibility -emit-llvm -o - %s

// expected-no-diagnostics

#ifdef _WIN64
#define ATTR(X) __declspec(X)
#else
#define ATTR(X) __attribute__((X))
#endif // _WIN64

ATTR(cpu_specific(generic)) void CPU(void){}
ATTR(cpu_specific(pentium)) void CPU(void){}
ATTR(cpu_specific(pentium_pro)) void CPU(void){}
ATTR(cpu_specific(pentium_mmx)) void CPU(void){}
ATTR(cpu_specific(pentium_ii)) void CPU(void){}
ATTR(cpu_specific(pentium_iii)) void CPU(void){}
ATTR(cpu_specific(pentium_4)) void CPU(void){}
ATTR(cpu_specific(pentium_m)) void CPU(void){}
ATTR(cpu_specific(pentium_4_sse3)) void CPU(void){}
ATTR(cpu_specific(core_2_duo_ssse3)) void CPU(void){}
ATTR(cpu_specific(core_2_duo_sse4_1)) void CPU(void){}
ATTR(cpu_specific(atom)) void CPU(void){}
ATTR(cpu_specific(atom_sse4_2)) void CPU(void){}
ATTR(cpu_specific(core_i7_sse4_2)) void CPU(void){}
ATTR(cpu_specific(core_aes_pclmulqdq)) void CPU(void){}
ATTR(cpu_specific(atom_sse4_2_movbe)) void CPU(void){}
ATTR(cpu_specific(goldmont)) void CPU(void){}
ATTR(cpu_specific(sandybridge)) void CPU(void){}
ATTR(cpu_specific(ivybridge)) void CPU(void){}
ATTR(cpu_specific(haswell)) void CPU(void){}
ATTR(cpu_specific(core_4th_gen_avx_tsx)) void CPU(void){}
ATTR(cpu_specific(broadwell)) void CPU(void){}
ATTR(cpu_specific(core_5th_gen_avx_tsx)) void CPU(void){}
ATTR(cpu_specific(knl)) void CPU(void){}
ATTR(cpu_specific(skylake)) void CPU(void){}
ATTR(cpu_specific(skylake_avx512)) void CPU(void){}
ATTR(cpu_specific(cannonlake)) void CPU(void){}
ATTR(cpu_specific(knm)) void CPU(void){}

// ALIAS CPUs
ATTR(cpu_specific(pentium_iii_no_xmm_regs)) void CPU0(void){}
ATTR(cpu_specific(core_2nd_gen_avx)) void CPU1(void){}
ATTR(cpu_specific(core_3rd_gen_avx)) void CPU2(void){}
ATTR(cpu_specific(core_4th_gen_avx)) void CPU3(void){}
ATTR(cpu_specific(core_5th_gen_avx)) void CPU4(void){}
ATTR(cpu_specific(mic_avx512)) void CPU5(void){}
