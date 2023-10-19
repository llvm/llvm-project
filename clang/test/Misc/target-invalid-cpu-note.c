// Use CHECK-NEXT instead of multiple CHECK-SAME to ensure we will fail if there is anything extra in the output.
// RUN: not %clang_cc1 -triple armv5--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix ARM
// ARM: error: unknown target CPU 'not-a-cpu'
// ARM-NEXT: note: valid target CPU values are: arm8, arm810, strongarm, strongarm110, strongarm1100, strongarm1110, arm7tdmi, arm7tdmi-s, arm710t, arm720t, arm9, arm9tdmi, arm920, arm920t, arm922t, arm940t, ep9312, arm10tdmi, arm1020t, arm9e, arm946e-s, arm966e-s, arm968e-s, arm10e, arm1020e, arm1022e, arm926ej-s, arm1136j-s, arm1136jf-s, mpcore, mpcorenovfp, arm1176jz-s, arm1176jzf-s, arm1156t2-s, arm1156t2f-s, cortex-m0, cortex-m0plus, cortex-m1, sc000, cortex-a5, cortex-a7, cortex-a8, cortex-a9, cortex-a12, cortex-a15, cortex-a17, krait, cortex-r4, cortex-r4f, cortex-r5, cortex-r7, cortex-r8, cortex-r52, sc300, cortex-m3, cortex-m4, cortex-m7, cortex-m23, cortex-m33, cortex-m35p, cortex-m55, cortex-m85, cortex-a32, cortex-a35, cortex-a53, cortex-a55, cortex-a57, cortex-a72, cortex-a73, cortex-a75, cortex-a76, cortex-a76ae, cortex-a77, cortex-a78, cortex-a78c, cortex-a710, cortex-x1, cortex-x1c, neoverse-n1, neoverse-n2, neoverse-v1, cyclone, exynos-m3, exynos-m4, exynos-m5, kryo, iwmmxt, xscale, swift{{$}}

// RUN: not %clang_cc1 -triple arm64--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix AARCH64
// AARCH64: error: unknown target CPU 'not-a-cpu'
// AARCH64-NEXT: note: valid target CPU values are: cortex-a34, cortex-a35, cortex-a53, cortex-a55, cortex-a510, cortex-a57, cortex-a65, cortex-a65ae, cortex-a72, cortex-a73, cortex-a75, cortex-a76, cortex-a76ae, cortex-a77, cortex-a78, cortex-a78c, cortex-a710, cortex-a715, cortex-r82, cortex-x1, cortex-x1c, cortex-x2, cortex-x3, neoverse-e1, neoverse-n1, neoverse-n2, neoverse-512tvb, neoverse-v1, neoverse-v2, cyclone, apple-a7, apple-a8, apple-a9, apple-a10, apple-a11, apple-a12, apple-a13, apple-a14, apple-a15, apple-a16, apple-m1, apple-m2, apple-s4, apple-s5, exynos-m3, exynos-m4, exynos-m5, falkor, saphira, kryo, thunderx2t99, thunderx3t110, thunderx, thunderxt88, thunderxt81, thunderxt83, tsv110, a64fx, carmel, ampere1, ampere1a, grace{{$}}

// RUN: not %clang_cc1 -triple arm64--- -tune-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix TUNE_AARCH64
// TUNE_AARCH64: error: unknown target CPU 'not-a-cpu'
// TUNE_AARCH64-NEXT: note: valid target CPU values are: cortex-a34, cortex-a35, cortex-a53, cortex-a55, cortex-a510, cortex-a57, cortex-a65, cortex-a65ae, cortex-a72, cortex-a73, cortex-a75, cortex-a76, cortex-a76ae, cortex-a77, cortex-a78, cortex-a78c, cortex-a710, cortex-a715, cortex-r82, cortex-x1, cortex-x1c, cortex-x2, cortex-x3, neoverse-e1, neoverse-n1, neoverse-n2, neoverse-512tvb, neoverse-v1, neoverse-v2, cyclone, apple-a7, apple-a8, apple-a9, apple-a10, apple-a11, apple-a12, apple-a13, apple-a14, apple-a15, apple-a16,  apple-m1, apple-m2, apple-s4, apple-s5, exynos-m3, exynos-m4, exynos-m5, falkor, saphira, kryo, thunderx2t99, thunderx3t110, thunderx, thunderxt88, thunderxt81, thunderxt83, tsv110, a64fx, carmel, ampere1, ampere1a, grace{{$}}

// RUN: not %clang_cc1 -triple i386--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix X86
// X86: error: unknown target CPU 'not-a-cpu'
// X86-NEXT: note: valid target CPU values are: i386, i486, winchip-c6, winchip2, c3, i586, pentium, pentium-mmx, pentiumpro, i686, pentium2, pentium3, pentium3m, pentium-m, c3-2, yonah, pentium4, pentium4m, prescott, nocona, core2, penryn, bonnell, atom, silvermont, slm, goldmont, goldmont-plus, tremont, nehalem, corei7, westmere, sandybridge, corei7-avx, ivybridge, core-avx-i, haswell, core-avx2, broadwell, skylake, skylake-avx512, skx, cascadelake, cooperlake, cannonlake, icelake-client, rocketlake, icelake-server, tigerlake, sapphirerapids, alderlake, raptorlake, meteorlake, arrowlake, arrowlake-s, lunarlake, gracemont, pantherlake, sierraforest, grandridge, graniterapids, graniterapids-d, emeraldrapids, clearwaterforest, knl, knm, lakemont, k6, k6-2, k6-3, athlon, athlon-tbird, athlon-xp, athlon-mp, athlon-4, k8, athlon64, athlon-fx, opteron, k8-sse3, athlon64-sse3, opteron-sse3, amdfam10, barcelona, btver1, btver2, bdver1, bdver2, bdver3, bdver4, znver1, znver2, znver3, znver4, x86-64, x86-64-v2, x86-64-v3, x86-64-v4, geode{{$}}

// RUN: not %clang_cc1 -triple x86_64--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix X86_64
// X86_64: error: unknown target CPU 'not-a-cpu'
// X86_64-NEXT: note: valid target CPU values are: nocona, core2, penryn, bonnell, atom, silvermont, slm, goldmont, goldmont-plus, tremont, nehalem, corei7, westmere, sandybridge, corei7-avx, ivybridge, core-avx-i, haswell, core-avx2, broadwell, skylake, skylake-avx512, skx, cascadelake, cooperlake, cannonlake, icelake-client, rocketlake, icelake-server, tigerlake, sapphirerapids, alderlake, raptorlake, meteorlake, arrowlake, arrowlake-s, lunarlake, gracemont, pantherlake, sierraforest, grandridge, graniterapids, graniterapids-d, emeraldrapids, clearwaterforest, knl, knm, k8, athlon64, athlon-fx, opteron, k8-sse3, athlon64-sse3, opteron-sse3, amdfam10, barcelona, btver1, btver2, bdver1, bdver2, bdver3, bdver4, znver1, znver2, znver3, znver4, x86-64, x86-64-v2, x86-64-v3, x86-64-v4{{$}}

// RUN: not %clang_cc1 -triple i386--- -tune-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix TUNE_X86
// TUNE_X86: error: unknown target CPU 'not-a-cpu'
// TUNE_X86-NEXT: note: valid target CPU values are: i386, i486, winchip-c6, winchip2, c3, i586, pentium, pentium-mmx, pentiumpro, i686, pentium2, pentium3, pentium3m, pentium-m, c3-2, yonah, pentium4, pentium4m, prescott, nocona, core2, penryn, bonnell, atom, silvermont, slm, goldmont, goldmont-plus, tremont, nehalem, corei7, westmere, sandybridge, corei7-avx, ivybridge, core-avx-i, haswell, core-avx2, broadwell, skylake, skylake-avx512, skx, cascadelake, cooperlake, cannonlake, icelake-client, rocketlake, icelake-server, tigerlake, sapphirerapids, alderlake, raptorlake, meteorlake, arrowlake, arrowlake-s, lunarlake, gracemont, pantherlake, sierraforest, grandridge, graniterapids, graniterapids-d, emeraldrapids, clearwaterforest, knl, knm, lakemont, k6, k6-2, k6-3, athlon, athlon-tbird, athlon-xp, athlon-mp, athlon-4, k8, athlon64, athlon-fx, opteron, k8-sse3, athlon64-sse3, opteron-sse3, amdfam10, barcelona, btver1, btver2, bdver1, bdver2, bdver3, bdver4, znver1, znver2, znver3, znver4, x86-64, geode{{$}}

// RUN: not %clang_cc1 -triple x86_64--- -tune-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix TUNE_X86_64
// TUNE_X86_64: error: unknown target CPU 'not-a-cpu'
// TUNE_X86_64-NEXT: note: valid target CPU values are: i386, i486, winchip-c6, winchip2, c3, i586, pentium, pentium-mmx, pentiumpro, i686, pentium2, pentium3, pentium3m, pentium-m, c3-2, yonah, pentium4, pentium4m, prescott, nocona, core2, penryn, bonnell, atom, silvermont, slm, goldmont, goldmont-plus, tremont, nehalem, corei7, westmere, sandybridge, corei7-avx, ivybridge, core-avx-i, haswell, core-avx2, broadwell, skylake, skylake-avx512, skx, cascadelake, cooperlake, cannonlake, icelake-client, rocketlake, icelake-server, tigerlake, sapphirerapids, alderlake, raptorlake, meteorlake, arrowlake, arrowlake-s, lunarlake, gracemont, pantherlake, sierraforest, grandridge, graniterapids, graniterapids-d, emeraldrapids, clearwaterforest, knl, knm, lakemont, k6, k6-2, k6-3, athlon, athlon-tbird, athlon-xp, athlon-mp, athlon-4, k8, athlon64, athlon-fx, opteron, k8-sse3, athlon64-sse3, opteron-sse3, amdfam10, barcelona, btver1, btver2, bdver1, bdver2, bdver3, bdver4, znver1, znver2, znver3, znver4, x86-64, geode{{$}}

// RUN: not %clang_cc1 -triple nvptx--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix NVPTX
// NVPTX: error: unknown target CPU 'not-a-cpu'
// NVPTX-NEXT: note: valid target CPU values are: sm_20, sm_21, sm_30, sm_32, sm_35, sm_37, sm_50, sm_52, sm_53, sm_60, sm_61, sm_62, sm_70, sm_72, sm_75, sm_80, sm_86, sm_87, sm_89, sm_90, gfx600, gfx601, gfx602, gfx700, gfx701, gfx702, gfx703, gfx704, gfx705, gfx801, gfx802, gfx803, gfx805, gfx810, gfx900, gfx902, gfx904, gfx906, gfx908, gfx909, gfx90a, gfx90c, gfx940, gfx941, gfx942, gfx1010, gfx1011, gfx1012, gfx1013, gfx1030, gfx1031, gfx1032, gfx1033, gfx1034, gfx1035, gfx1036, gfx1100, gfx1101, gfx1102, gfx1103, gfx1105, gfx1150, gfx1151{{$}}

// RUN: not %clang_cc1 -triple r600--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix R600
// R600: error: unknown target CPU 'not-a-cpu'
// R600-NEXT: note: valid target CPU values are: r600, rv630, rv635, r630, rs780, rs880, rv610, rv620, rv670, rv710, rv730, rv740, rv770, cedar, palm, cypress, hemlock, juniper, redwood, sumo, sumo2, barts, caicos, aruba, cayman, turks{{$}}

// RUN: not %clang_cc1 -triple amdgcn--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix AMDGCN
// AMDGCN: error: unknown target CPU 'not-a-cpu'
// AMDGCN-NEXT: note: valid target CPU values are: gfx600, tahiti, gfx601, pitcairn, verde, gfx602, hainan, oland, gfx700, kaveri, gfx701, hawaii, gfx702, gfx703, kabini, mullins, gfx704, bonaire, gfx705, gfx801, carrizo, gfx802, iceland, tonga, gfx803, fiji, polaris10, polaris11, gfx805, tongapro, gfx810, stoney, gfx900, gfx902, gfx904, gfx906, gfx908, gfx909, gfx90a, gfx90c, gfx940, gfx941, gfx942, gfx1010, gfx1011, gfx1012, gfx1013, gfx1030, gfx1031, gfx1032, gfx1033, gfx1034, gfx1035, gfx1036, gfx1100, gfx1101, gfx1102, gfx1103, gfx1105, gfx1150, gfx1151{{$}}

// RUN: not %clang_cc1 -triple wasm64--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix WEBASM
// WEBASM: error: unknown target CPU 'not-a-cpu'
// WEBASM-NEXT: note: valid target CPU values are: mvp, bleeding-edge, generic{{$}}

// RUN: not %clang_cc1 -triple systemz--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix SYSTEMZ
// SYSTEMZ: error: unknown target CPU 'not-a-cpu'
// SYSTEMZ-NEXT: note: valid target CPU values are: arch8, z10, arch9, z196, arch10, zEC12, arch11, z13, arch12, z14, arch13, z15, arch14, z16{{$}}

// RUN: not %clang_cc1 -triple sparc--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix SPARC
// SPARC: error: unknown target CPU 'not-a-cpu'
// SPARC-NEXT: note: valid target CPU values are: v8, supersparc, sparclite, f934, hypersparc, sparclite86x, sparclet, tsc701, v9, ultrasparc, ultrasparc3, niagara, niagara2, niagara3, niagara4, ma2100, ma2150, ma2155, ma2450, ma2455, ma2x5x, ma2080, ma2085, ma2480, ma2485, ma2x8x, leon2, at697e, at697f, leon3, ut699, gr712rc, leon4, gr740{{$}}

// RUN: not %clang_cc1 -triple sparcv9--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix SPARCV9
// SPARCV9: error: unknown target CPU 'not-a-cpu'
// SPARCV9-NEXT: note: valid target CPU values are: v9, ultrasparc, ultrasparc3, niagara, niagara2, niagara3, niagara4{{$}}

// RUN: not %clang_cc1 -triple powerpc--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix PPC
// PPC: error: unknown target CPU 'not-a-cpu'
// PPC-NEXT: note: valid target CPU values are: generic, 440, 450, 601, 602, 603, 603e, 603ev, 604, 604e, 620, 630, g3, 7400, g4, 7450, g4+, 750, 8548, 970, g5, a2, e500, e500mc, e5500, power3, pwr3, power4, pwr4, power5, pwr5, power5x, pwr5x, power6, pwr6, power6x, pwr6x, power7, pwr7, power8, pwr8, power9, pwr9, power10, pwr10, powerpc, ppc, ppc32, powerpc64, ppc64, powerpc64le, ppc64le, future{{$}}

// RUN: not %clang_cc1 -triple mips--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix MIPS
// MIPS: error: unknown target CPU 'not-a-cpu'
// MIPS-NEXT: note: valid target CPU values are: mips1, mips2, mips3, mips4, mips5, mips32, mips32r2, mips32r3, mips32r5, mips32r6, mips64, mips64r2, mips64r3, mips64r5, mips64r6, octeon, octeon+, p5600{{$}}

// RUN: not %clang_cc1 -triple lanai--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix LANAI
// LANAI: error: unknown target CPU 'not-a-cpu'
// LANAI-NEXT: note: valid target CPU values are: v11{{$}}

// RUN: not %clang_cc1 -triple hexagon--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix HEXAGON
// HEXAGON: error: unknown target CPU 'not-a-cpu'
// HEXAGON-NEXT: note: valid target CPU values are: hexagonv5, hexagonv55, hexagonv60, hexagonv62, hexagonv65, hexagonv66, hexagonv67, hexagonv67t, hexagonv68, hexagonv69, hexagonv71, hexagonv71t, hexagonv73{{$}}

// RUN: not %clang_cc1 -triple bpf--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix BPF
// BPF: error: unknown target CPU 'not-a-cpu'
// BPF-NEXT: note: valid target CPU values are: generic, v1, v2, v3, v4, probe{{$}}

// RUN: not %clang_cc1 -triple avr--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix AVR
// AVR: error: unknown target CPU 'not-a-cpu'
// AVR-NEXT: note: valid target CPU values are: avr1, at90s1200, attiny11, attiny12, attiny15, attiny28, avr2, at90s2313, at90s2323, at90s2333, at90s2343, attiny22, attiny26, at86rf401, at90s4414, at90s4433, at90s4434, at90s8515, at90c8534, at90s8535, avr25, ata5272, ata6616c, attiny13, attiny13a, attiny2313, attiny2313a, attiny24, attiny24a, attiny4313, attiny44, attiny44a, attiny84, attiny84a, attiny25, attiny45, attiny85, attiny261, attiny261a, attiny441, attiny461, attiny461a, attiny841, attiny861, attiny861a, attiny87, attiny43u, attiny48, attiny88, attiny828, avr3, at43usb355, at76c711, avr31, atmega103, at43usb320, avr35, attiny167, at90usb82, at90usb162, ata5505, ata6617c, ata664251, atmega8u2, atmega16u2, atmega32u2, attiny1634, avr4, atmega8, ata6289, atmega8a, ata6285, ata6286, ata6612c, atmega48, atmega48a, atmega48pa, atmega48pb, atmega48p, atmega88, atmega88a, atmega88p, atmega88pa, atmega88pb, atmega8515, atmega8535, atmega8hva, at90pwm1, at90pwm2, at90pwm2b, at90pwm3, at90pwm3b, at90pwm81, avr5, ata5702m322, ata5782, ata5790, ata5790n, ata5791, ata5795, ata5831, ata6613c, ata6614q, ata8210, ata8510, atmega16, atmega16a, atmega161, atmega162, atmega163, atmega164a, atmega164p, atmega164pa, atmega165, atmega165a, atmega165p, atmega165pa, atmega168, atmega168a, atmega168p, atmega168pa, atmega168pb, atmega169, atmega169a, atmega169p, atmega169pa, atmega32, atmega32a, atmega323, atmega324a, atmega324p, atmega324pa, atmega324pb, atmega325, atmega325a, atmega325p, atmega325pa, atmega3250, atmega3250a, atmega3250p, atmega3250pa, atmega328, atmega328p, atmega328pb, atmega329, atmega329a, atmega329p, atmega329pa, atmega3290, atmega3290a, atmega3290p, atmega3290pa, atmega406, atmega64, atmega64a, atmega640, atmega644, atmega644a, atmega644p, atmega644pa, atmega645, atmega645a, atmega645p, atmega649, atmega649a, atmega649p, atmega6450, atmega6450a, atmega6450p, atmega6490, atmega6490a, atmega6490p, atmega64rfr2, atmega644rfr2, atmega16hva, atmega16hva2, atmega16hvb, atmega16hvbrevb, atmega32hvb, atmega32hvbrevb, atmega64hve, atmega64hve2, at90can32, at90can64, at90pwm161, at90pwm216, at90pwm316, atmega32c1, atmega64c1, atmega16m1, atmega32m1, atmega64m1, atmega16u4, atmega32u4, atmega32u6, at90usb646, at90usb647, at90scr100, at94k, m3000, avr51, atmega128, atmega128a, atmega1280, atmega1281, atmega1284, atmega1284p, atmega128rfa1, atmega128rfr2, atmega1284rfr2, at90can128, at90usb1286, at90usb1287, avr6, atmega2560, atmega2561, atmega256rfr2, atmega2564rfr2, avrxmega2, atxmega16a4, atxmega16a4u, atxmega16c4, atxmega16d4, atxmega32a4, atxmega32a4u, atxmega32c3, atxmega32c4, atxmega32d3, atxmega32d4, atxmega32e5, atxmega16e5, atxmega8e5, avrxmega4, atxmega64a3, atxmega64a3u, atxmega64a4u, atxmega64b1, atxmega64b3, atxmega64c3, atxmega64d3, atxmega64d4, avrxmega5, atxmega64a1, atxmega64a1u, avrxmega6, atxmega128a3, atxmega128a3u, atxmega128b1, atxmega128b3, atxmega128c3, atxmega128d3, atxmega128d4, atxmega192a3, atxmega192a3u, atxmega192c3, atxmega192d3, atxmega256a3, atxmega256a3u, atxmega256a3b, atxmega256a3bu, atxmega256c3, atxmega256d3, atxmega384c3, atxmega384d3, avrxmega7, atxmega128a1, atxmega128a1u, atxmega128a4u, avrtiny, attiny4, attiny5, attiny9, attiny10, attiny20, attiny40, attiny102, attiny104, avrxmega3, attiny202, attiny402, attiny204, attiny404, attiny804, attiny1604, attiny406, attiny806, attiny1606, attiny807, attiny1607, attiny212, attiny412, attiny214, attiny414, attiny814, attiny1614, attiny416, attiny816, attiny1616, attiny3216, attiny417, attiny817, attiny1617, attiny3217, attiny1624, attiny1626, attiny1627, atmega808, atmega809, atmega1608, atmega1609, atmega3208, atmega3209, atmega4808, atmega4809

// RUN: not %clang_cc1 -triple riscv32 -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix RISCV32
// RISCV32: error: unknown target CPU 'not-a-cpu'
// RISCV32-NEXT: note: valid target CPU values are: generic-rv32, rocket-rv32, sifive-e20, sifive-e21, sifive-e24, sifive-e31, sifive-e34, sifive-e76, syntacore-scr1-base, syntacore-scr1-max{{$}}

// RUN: not %clang_cc1 -triple riscv64 -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix RISCV64
// RISCV64: error: unknown target CPU 'not-a-cpu'
// RISCV64-NEXT: note: valid target CPU values are: generic-rv64, rocket-rv64, sifive-s21, sifive-s51, sifive-s54, sifive-s76, sifive-u54, sifive-u74, sifive-x280, veyron-v1{{$}}

// RUN: not %clang_cc1 -triple riscv32 -tune-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix TUNE-RISCV32
// TUNE-RISCV32: error: unknown target CPU 'not-a-cpu'
// TUNE-RISCV32-NEXT: note: valid target CPU values are: generic-rv32, rocket-rv32, sifive-e20, sifive-e21, sifive-e24, sifive-e31, sifive-e34, sifive-e76, syntacore-scr1-base, syntacore-scr1-max, generic, rocket, sifive-7-series{{$}}

// RUN: not %clang_cc1 -triple riscv64 -tune-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix TUNE-RISCV64
// TUNE-RISCV64: error: unknown target CPU 'not-a-cpu'
// TUNE-RISCV64-NEXT: note: valid target CPU values are: generic-rv64, rocket-rv64, sifive-s21, sifive-s51, sifive-s54, sifive-s76, sifive-u54, sifive-u74, sifive-x280, veyron-v1, generic, rocket, sifive-7-series{{$}}
