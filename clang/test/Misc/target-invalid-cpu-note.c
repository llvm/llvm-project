// RUN: not %clang_cc1 -triple armv5--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix ARM
// ARM: error: unknown target CPU 'not-a-cpu'
// ARM: note: valid target CPU values are:
// ARM-SAME: arm2

// RUN: not %clang_cc1 -triple arm64--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix AARCH64
// AARCH64: error: unknown target CPU 'not-a-cpu'
// AARCH64: note: valid target CPU values are:
// AARCH64-SAME: cortex-a35,

// RUN: not %clang_cc1 -triple arm64--- -tune-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix TUNE_AARCH64
// TUNE_AARCH64: error: unknown target CPU 'not-a-cpu'
// TUNE_AARCH64: note: valid target CPU values are:
// TUNE_AARCH64-SAME: cortex-a35,

// RUN: not %clang_cc1 -triple i386--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix X86
// X86: error: unknown target CPU 'not-a-cpu'
// X86: note: valid target CPU values are: i386, i486, winchip-c6, winchip2, c3,
// X86-SAME: i586, pentium, pentium-mmx, pentiumpro, i686, pentium2, pentium3,
// X86-SAME: pentium3m, pentium-m, c3-2, yonah, pentium4, pentium4m, prescott,
// X86-SAME: nocona, core2, penryn, bonnell, atom, silvermont, slm, goldmont, goldmont-plus, tremont,
// X86-SAME: nehalem, corei7, westmere, sandybridge, corei7-avx, ivybridge,
// X86-SAME: core-avx-i, haswell, core-avx2, broadwell, skylake, skylake-avx512,
// X86-SAME: skx, cascadelake, cooperlake, cannonlake, icelake-client, icelake-server, tigerlake, sapphirerapids, alderlake, knl, knm, lakemont, k6, k6-2, k6-3,
// X86-SAME: athlon, athlon-tbird, athlon-xp, athlon-mp, athlon-4, k8, athlon64,
// X86-SAME: athlon-fx, opteron, k8-sse3, athlon64-sse3, opteron-sse3, amdfam10,
// X86-SAME: barcelona, btver1, btver2, bdver1, bdver2, bdver3, bdver4, znver1, znver2, znver3,
// X86-SAME: x86-64, x86-64-v2, x86-64-v3, x86-64-v4, geode{{$}}

// RUN: not %clang_cc1 -triple x86_64--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix X86_64
// X86_64: error: unknown target CPU 'not-a-cpu'
// X86_64: note: valid target CPU values are: nocona, core2, penryn, bonnell,
// X86_64-SAME: atom, silvermont, slm, goldmont, goldmont-plus, tremont, nehalem, corei7, westmere,
// X86_64-SAME: sandybridge, corei7-avx, ivybridge, core-avx-i, haswell,
// X86_64-SAME: core-avx2, broadwell, skylake, skylake-avx512, skx, cascadelake, cooperlake, cannonlake,
// X86_64-SAME: icelake-client, icelake-server, tigerlake, sapphirerapids, alderlake, knl, knm, k8, athlon64, athlon-fx, opteron, k8-sse3,
// X86_64-SAME: athlon64-sse3, opteron-sse3, amdfam10, barcelona, btver1,
// X86_64-SAME: btver2, bdver1, bdver2, bdver3, bdver4, znver1, znver2, znver3,
// X86_64-SAME: x86-64, x86-64-v2, x86-64-v3, x86-64-v4{{$}}

// RUN: not %clang_cc1 -triple i386--- -tune-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix TUNE_X86
// TUNE_X86: error: unknown target CPU 'not-a-cpu'
// TUNE_X86: note: valid target CPU values are: i386, i486, winchip-c6, winchip2, c3,
// TUNE_X86-SAME: i586, pentium, pentium-mmx, pentiumpro, i686, pentium2, pentium3,
// TUNE_X86-SAME: pentium3m, pentium-m, c3-2, yonah, pentium4, pentium4m, prescott,
// TUNE_X86-SAME: nocona, core2, penryn, bonnell, atom, silvermont, slm, goldmont, goldmont-plus, tremont,
// TUNE_X86-SAME: nehalem, corei7, westmere, sandybridge, corei7-avx, ivybridge,
// TUNE_X86-SAME: core-avx-i, haswell, core-avx2, broadwell, skylake, skylake-avx512,
// TUNE_X86-SAME: skx, cascadelake, cooperlake, cannonlake, icelake-client, icelake-server, tigerlake, sapphirerapids, alderlake, knl, knm, lakemont, k6, k6-2, k6-3,
// TUNE_X86-SAME: athlon, athlon-tbird, athlon-xp, athlon-mp, athlon-4, k8, athlon64,
// TUNE_X86-SAME: athlon-fx, opteron, k8-sse3, athlon64-sse3, opteron-sse3, amdfam10,
// TUNE_X86-SAME: barcelona, btver1, btver2, bdver1, bdver2, bdver3, bdver4, znver1, znver2, znver3,
// TUNE_X86-SAME: x86-64, geode{{$}}

// RUN: not %clang_cc1 -triple x86_64--- -tune-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix TUNE_X86_64
// TUNE_X86_64: error: unknown target CPU 'not-a-cpu'
// TUNE_X86_64: note: valid target CPU values are: i386, i486, winchip-c6, winchip2, c3,
// TUNE_X86_64-SAME: i586, pentium, pentium-mmx, pentiumpro, i686, pentium2, pentium3,
// TUNE_X86_64-SAME: pentium3m, pentium-m, c3-2, yonah, pentium4, pentium4m, prescott,
// TUNE_X86_64-SAME: nocona, core2, penryn, bonnell, atom, silvermont, slm, goldmont, goldmont-plus, tremont,
// TUNE_X86_64-SAME: nehalem, corei7, westmere, sandybridge, corei7-avx, ivybridge,
// TUNE_X86_64-SAME: core-avx-i, haswell, core-avx2, broadwell, skylake, skylake-avx512,
// TUNE_X86_64-SAME: skx, cascadelake, cooperlake, cannonlake, icelake-client, icelake-server, tigerlake, sapphirerapids, alderlake, knl, knm, lakemont, k6, k6-2, k6-3,
// TUNE_X86_64-SAME: athlon, athlon-tbird, athlon-xp, athlon-mp, athlon-4, k8, athlon64,
// TUNE_X86_64-SAME: athlon-fx, opteron, k8-sse3, athlon64-sse3, opteron-sse3, amdfam10,
// TUNE_X86_64-SAME: barcelona, btver1, btver2, bdver1, bdver2, bdver3, bdver4, znver1, znver2, znver3,
// TUNE_X86_64-SAME: x86-64, geode{{$}}

// RUN: not %clang_cc1 -triple nvptx--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix NVPTX
// NVPTX: error: unknown target CPU 'not-a-cpu'
// NVPTX: note: valid target CPU values are: sm_20, sm_21, sm_30, sm_32, sm_35,
// NVPTX-SAME: sm_37, sm_50, sm_52, sm_53, sm_60, sm_61, sm_62, sm_70, sm_72

// RUN: not %clang_cc1 -triple r600--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix R600
// R600: error: unknown target CPU 'not-a-cpu'
// R600: note: valid target CPU values are: r600, rv630, rv635, r630, rs780, 
// R600-SAME: rs880, rv610, rv620, rv670, rv710, rv730, rv740, rv770, cedar, 
// R600-SAME: palm, cypress, hemlock, juniper, redwood, sumo, sumo2, barts, 
// R600-SAME: caicos, aruba, cayman, turks


// RUN: not %clang_cc1 -triple amdgcn--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix AMDGCN
// AMDGCN: error: unknown target CPU 'not-a-cpu'
// AMDGCN: note: valid target CPU values are: gfx600, tahiti, gfx601, pitcairn, verde,
// AMDGCN-SAME: gfx602, hainan, oland, gfx700, kaveri, gfx701, hawaii, gfx702,
// AMDGCN-SAME: gfx703, kabini, mullins, gfx704, bonaire, gfx705, gfx801, carrizo, 
// AMDGCN-SAME: gfx802, iceland, tonga, gfx803, fiji, polaris10, polaris11,
// AMDGCN-SAME: gfx805, tongapro, gfx810, stoney, gfx900, gfx902

// RUN: not %clang_cc1 -triple wasm64--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix WEBASM
// WEBASM: error: unknown target CPU 'not-a-cpu'
// WEBASM: note: valid target CPU values are: mvp, bleeding-edge, generic

// RUN: not %clang_cc1 -triple systemz--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix SYSTEMZ
// SYSTEMZ: error: unknown target CPU 'not-a-cpu'
// SYSTEMZ: note: valid target CPU values are: arch8, z10, arch9, z196, arch10,
// SYSTEMZ-SAME: zEC12, arch11, z13, arch12, z14, arch13, z15

// RUN: not %clang_cc1 -triple sparc--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix SPARC
// SPARC: error: unknown target CPU 'not-a-cpu'
// SPARC: note: valid target CPU values are: v8, supersparc, sparclite, f934,
// SPARC-SAME: hypersparc, sparclite86x, sparclet, tsc701, v9, ultrasparc,
// SPARC-SAME: ultrasparc3, niagara, niagara2, niagara3, niagara4, ma2100,
// SPARC-SAME: ma2150, ma2155, ma2450, ma2455, ma2x5x, ma2080, ma2085, ma2480,
// SPARC-SAME: ma2485, ma2x8x, myriad2, myriad2.1, myriad2.2, myriad2.3, leon2,
// SPARC-SAME: at697e, at697f, leon3, ut699, gr712rc, leon4, gr740

// RUN: not %clang_cc1 -triple sparcv9--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix SPARCV9
// SPARCV9: error: unknown target CPU 'not-a-cpu'
// SPARCV9: note: valid target CPU values are: v9, ultrasparc, ultrasparc3, niagara, niagara2, niagara3, niagara4

// RUN: not %clang_cc1 -triple powerpc--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix PPC
// PPC: error: unknown target CPU 'not-a-cpu'
// PPC: note: valid target CPU values are: generic, 440, 450, 601, 602, 603,
// PPC-SAME: 603e, 603ev, 604, 604e, 620, 630, g3, 7400, g4, 7450, g4+, 750,
// PPC-SAME: 8548, 970, g5, a2, e500, e500mc, e5500, power3, pwr3, power4,
// PPC-SAME: pwr4, power5, pwr5, power5x, pwr5x, power6, pwr6, power6x, pwr6x,
// PPC-SAME: power7, pwr7, power8, pwr8, power9, pwr9, power10, pwr10, powerpc, ppc, powerpc64,
// PPC-SAME: ppc64, powerpc64le, ppc64le, future

// RUN: not %clang_cc1 -triple mips--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix MIPS
// MIPS: error: unknown target CPU 'not-a-cpu'
// MIPS: note: valid target CPU values are: mips1, mips2, mips3, mips4, mips5,
// MIPS-SAME: mips32, mips32r2, mips32r3, mips32r5, mips32r6, mips64, mips64r2,
// MIPS-SAME: mips64r3, mips64r5, mips64r6, octeon, octeon+, p5600

// RUN: not %clang_cc1 -triple lanai--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix LANAI
// LANAI: error: unknown target CPU 'not-a-cpu'
// LANAI: note: valid target CPU values are: v11

// RUN: not %clang_cc1 -triple hexagon--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix HEXAGON
// HEXAGON: error: unknown target CPU 'not-a-cpu'
// HEXAGON: note: valid target CPU values are: hexagonv5, hexagonv55,
// HEXAGON-SAME: hexagonv60, hexagonv62, hexagonv65

// RUN: not %clang_cc1 -triple bpf--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix BPF
// BPF: error: unknown target CPU 'not-a-cpu'
// BPF: note: valid target CPU values are: generic, v1, v2, v3, probe

// RUN: not %clang_cc1 -triple avr--- -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix AVR
// AVR: error: unknown target CPU 'not-a-cpu'
// AVR: note: valid target CPU values are: avr1, avr2, avr25, avr3, avr31,
// AVR-SAME: avr35, avr4, avr5, avr51, avr6, avrxmega1, avrxmega2, avrxmega3,
// AVR-SAME: avrxmega4, avrxmega5, avrxmega6, avrxmega7, avrtiny, at90s1200,
// AVR-SAME: attiny11, attiny12, attiny15, attiny28, at90s2313, at90s2323,
// AVR-SAME: at90s2333, at90s2343, attiny22, attiny26, at86rf401, at90s4414,
// AVR-SAME: at90s4433, at90s4434, at90s8515, at90c8534, at90s8535, ata5272,
// AVR-SAME: attiny13, attiny13a, attiny2313, attiny2313a, attiny24, attiny24a,
// AVR-SAME: attiny4313, attiny44, attiny44a, attiny84, attiny84a, attiny25,
// AVR-SAME: attiny45, attiny85, attiny261, attiny261a, attiny441, attiny461,
// AVR-SAME: attiny461a, attiny841, attiny861, attiny861a, attiny87, attiny43u,
// AVR-SAME: attiny48, attiny88, attiny828, at43usb355, at76c711, atmega103,
// AVR-SAME: at43usb320, attiny167, at90usb82, at90usb162, ata5505, atmega8u2,
// AVR-SAME: atmega16u2, atmega32u2, attiny1634, atmega8, ata6289, atmega8a,
// AVR-SAME: ata6285, ata6286, atmega48, atmega48a, atmega48pa, atmega48pb,
// AVR-SAME: atmega48p, atmega88, atmega88a, atmega88p, atmega88pa, atmega88pb,
// AVR-SAME: atmega8515, atmega8535, atmega8hva, at90pwm1, at90pwm2, at90pwm2b,
// AVR-SAME: at90pwm3, at90pwm3b, at90pwm81, ata5790, ata5795, atmega16,
// AVR-SAME: atmega16a, atmega161, atmega162, atmega163, atmega164a, atmega164p,
// AVR-SAME: atmega164pa, atmega165, atmega165a, atmega165p, atmega165pa,
// AVR-SAME: atmega168, atmega168a, atmega168p, atmega168pa, atmega168pb,
// AVR-SAME: atmega169, atmega169a, atmega169p, atmega169pa, atmega32, atmega32a,
// AVR-SAME: atmega323, atmega324a, atmega324p, atmega324pa, atmega324pb,
// AVR-SAME: atmega325, atmega325a, atmega325p, atmega325pa, atmega3250,
// AVR-SAME: atmega3250a, atmega3250p, atmega3250pa, atmega328, atmega328p,
// AVR-SAME: atmega328pb, atmega329, atmega329a, atmega329p, atmega329pa,
// AVR-SAME: atmega3290, atmega3290a, atmega3290p, atmega3290pa, atmega406,
// AVR-SAME: atmega64, atmega64a, atmega640, atmega644, atmega644a, atmega644p,
// AVR-SAME: atmega644pa, atmega645, atmega645a, atmega645p, atmega649, atmega649a,
// AVR-SAME: atmega649p, atmega6450, atmega6450a, atmega6450p, atmega6490,
// AVR-SAME: atmega6490a, atmega6490p, atmega64rfr2, atmega644rfr2, atmega16hva,
// AVR-SAME: atmega16hva2, atmega16hvb, atmega16hvbrevb, atmega32hvb,
// AVR-SAME: atmega32hvbrevb, atmega64hve, at90can32, at90can64, at90pwm161,
// AVR-SAME: at90pwm216, at90pwm316, atmega32c1, atmega64c1, atmega16m1, atmega32m1,
// AVR-SAME: atmega64m1, atmega16u4, atmega32u4, atmega32u6, at90usb646, at90usb647,
// AVR-SAME: at90scr100, at94k, m3000, atmega128, atmega128a, atmega1280, atmega1281,
// AVR-SAME: atmega1284, atmega1284p, atmega128rfa1, atmega128rfr2, atmega1284rfr2,
// AVR-SAME: at90can128, at90usb1286, at90usb1287, atmega2560, atmega2561,
// AVR-SAME: atmega256rfr2, atmega2564rfr2, atxmega16a4, atxmega16a4u, atxmega16c4,
// AVR-SAME: atxmega16d4, atxmega32a4, atxmega32a4u, atxmega32c4, atxmega32d4,
// AVR-SAME: atxmega32e5, atxmega16e5, atxmega8e5, atxmega32x1, atxmega64a3,
// AVR-SAME: atxmega64a3u, atxmega64a4u, atxmega64b1, atxmega64b3, atxmega64c3,
// AVR-SAME: atxmega64d3, atxmega64d4, atxmega64a1, atxmega64a1u, atxmega128a3,
// AVR-SAME: atxmega128a3u, atxmega128b1, atxmega128b3, atxmega128c3, atxmega128d3,
// AVR-SAME: atxmega128d4, atxmega192a3, atxmega192a3u, atxmega192c3, atxmega192d3,
// AVR-SAME: atxmega256a3, atxmega256a3u, atxmega256a3b, atxmega256a3bu, atxmega256c3,
// AVR-SAME: atxmega256d3, atxmega384c3, atxmega384d3, atxmega128a1, atxmega128a1u,
// AVR-SAME: atxmega128a4u, attiny4, attiny5, attiny9, attiny10, attiny20, attiny40,
// AVR-SAME: attiny102, attiny104

// RUN: not %clang_cc1 -triple riscv32 -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix RISCV32
// RISCV32: error: unknown target CPU 'not-a-cpu'
// RISCV32: note: valid target CPU values are: generic-rv32, rocket-rv32, sifive-7-rv32, sifive-e31, sifive-e76

// RUN: not %clang_cc1 -triple riscv64 -target-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix RISCV64
// RISCV64: error: unknown target CPU 'not-a-cpu'
// RISCV64: note: valid target CPU values are: generic-rv64, rocket-rv64, sifive-7-rv64, sifive-u54, sifive-u74

// RUN: not %clang_cc1 -triple riscv32 -tune-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix TUNE-RISCV32
// TUNE-RISCV32: error: unknown target CPU 'not-a-cpu'
// TUNE-RISCV32: note: valid target CPU values are: generic-rv32, rocket-rv32, sifive-7-rv32, sifive-e31, sifive-e76, generic, rocket, sifive-7-series

// RUN: not %clang_cc1 -triple riscv64 -tune-cpu not-a-cpu -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix TUNE-RISCV64
// TUNE-RISCV64: error: unknown target CPU 'not-a-cpu'
// TUNE-RISCV64: note: valid target CPU values are: generic-rv64, rocket-rv64, sifive-7-rv64, sifive-u54, sifive-u74, generic, rocket, sifive-7-series
