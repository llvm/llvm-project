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
ATTR(cpu_specific(cascadelake)) void CPU(void){}
ATTR(cpu_specific(cooperlake)) void CPU(void){}
ATTR(cpu_specific(icelake_client)) void CPU(void){}
ATTR(cpu_specific(tigerlake)) void CPU(void){}
ATTR(cpu_specific(alderlake)) void CPU(void){}
ATTR(cpu_specific(sapphirerapids)) void CPU(void){}
ATTR(cpu_specific(diamondrapids)) void CPU(void){}

// ALIAS CPUs
ATTR(cpu_specific(pentium_iii_no_xmm_regs)) void CPU0(void){}
ATTR(cpu_specific(core_2nd_gen_avx)) void CPU1(void){}
ATTR(cpu_specific(core_3rd_gen_avx)) void CPU2(void){}
ATTR(cpu_specific(core_4th_gen_avx)) void CPU3(void){}
ATTR(cpu_specific(core_5th_gen_avx)) void CPU4(void){}
ATTR(cpu_specific(mic_avx512)) void CPU5(void){}
ATTR(cpu_specific(pentiumpro)) void CPU6(void){}
ATTR(cpu_specific(pentium3)) void CPU7(void){}
ATTR(cpu_specific(pentium3m)) void CPU8(void){}
ATTR(cpu_specific(pentium4)) void CPU9(void){}
ATTR(cpu_specific(pentium4m)) void CPU10(void){}
ATTR(cpu_specific(yonah)) void CPU11(void){}
ATTR(cpu_specific(prescott)) void CPU12(void){}
ATTR(cpu_specific(nocona)) void CPU13(void){}
ATTR(cpu_specific(core2)) void CPU14(void){}
ATTR(cpu_specific(penryn)) void CPU15(void){}
ATTR(cpu_specific(bonnell)) void CPU16(void){}
ATTR(cpu_specific(silvermont)) void CPU17(void){}
ATTR(cpu_specific(slm)) void CPU18(void){}
ATTR(cpu_specific(goldmont_plus)) void CPU19(void){}
ATTR(cpu_specific(tremont)) void CPU20(void){}
ATTR(cpu_specific(nehalem)) void CPU21(void){}
ATTR(cpu_specific(corei7)) void CPU22(void){}
ATTR(cpu_specific(westmere)) void CPU23(void){}
ATTR(cpu_specific(sandybridge)) void CPU24(void){}
ATTR(cpu_specific(skx)) void CPU25(void){}
ATTR(cpu_specific(rocketlake)) void CPU26(void){}
ATTR(cpu_specific(icelake_server)) void CPU27(void){}
ATTR(cpu_specific(raptorlake)) void CPU28(void){}
ATTR(cpu_specific(meteorlake)) void CPU29(void){}
ATTR(cpu_specific(sierraforest)) void CPU30(void){}
ATTR(cpu_specific(grandridge)) void CPU31(void){}
ATTR(cpu_specific(graniterapids)) void CPU32(void){}
ATTR(cpu_specific(emeraldrapids)) void CPU33(void){}
ATTR(cpu_specific(graniterapids_d)) void CPU34(void){}
ATTR(cpu_specific(arrowlake)) void CPU35(void){}
ATTR(cpu_specific(arrowlake_s)) void CPU36(void){}
ATTR(cpu_specific(lunarlake)) void CPU37(void){}
ATTR(cpu_specific(gracemont)) void CPU38(void){}
ATTR(cpu_specific(pantherlake)) void CPU39(void){}
ATTR(cpu_specific(clearwaterforest)) void CPU40(void){}
