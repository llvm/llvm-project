// RUN: %clang -target i386-unknown-unknown -march=core2 -msse4 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SSE4 %s

// SSE4: #define __SSE2_MATH__ 1
// SSE4: #define __SSE2__ 1
// SSE4: #define __SSE3__ 1
// SSE4: #define __SSE4_1__ 1
// SSE4: #define __SSE4_2__ 1
// SSE4: #define __SSE_MATH__ 1
// SSE4: #define __SSE__ 1
// SSE4: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=core2 -msse4.1 -mno-sse4 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=NOSSE4 %s

// NOSSE4-NOT: #define __SSE4_1__ 1

// RUN: %clang -target i386-unknown-unknown -march=core2 -msse4 -mno-sse2 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SSE %s

// SSE-NOT: #define __SSE2_MATH__ 1
// SSE-NOT: #define __SSE2__ 1
// SSE-NOT: #define __SSE3__ 1
// SSE-NOT: #define __SSE4_1__ 1
// SSE-NOT: #define __SSE4_2__ 1
// SSE: #define __SSE_MATH__ 1
// SSE: #define __SSE__ 1
// SSE-NOT: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentium-m -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SSE2 %s

// SSE2: #define __SSE2_MATH__ 1
// SSE2: #define __SSE2__ 1
// SSE2-NOT: #define __SSE3__ 1
// SSE2-NOT: #define __SSE4_1__ 1
// SSE2-NOT: #define __SSE4_2__ 1
// SSE2: #define __SSE_MATH__ 1
// SSE2: #define __SSE__ 1
// SSE2-NOT: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentium-m -mno-sse -mavx -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX %s

// AVX: #define __AVX__ 1
// AVX: #define __SSE2_MATH__ 1
// AVX: #define __SSE2__ 1
// AVX: #define __SSE3__ 1
// AVX: #define __SSE4_1__ 1
// AVX: #define __SSE4_2__ 1
// AVX: #define __SSE_MATH__ 1
// AVX: #define __SSE__ 1
// AVX: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentium-m -mxop -mno-avx -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SSE4A %s

// SSE4A: #define __SSE2_MATH__ 1
// SSE4A: #define __SSE2__ 1
// SSE4A: #define __SSE3__ 1
// SSE4A: #define __SSE4A__ 1
// SSE4A: #define __SSE4_1__ 1
// SSE4A: #define __SSE4_2__ 1
// SSE4A: #define __SSE_MATH__ 1
// SSE4A: #define __SSE__ 1
// SSE4A: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512f -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512F %s

// AVX512F: #define __AVX2__ 1
// AVX512F: #define __AVX512F__ 1
// AVX512F: #define __AVX__ 1
// AVX512F: #define __SSE2_MATH__ 1
// AVX512F: #define __SSE2__ 1
// AVX512F: #define __SSE3__ 1
// AVX512F: #define __SSE4_1__ 1
// AVX512F: #define __SSE4_2__ 1
// AVX512F: #define __SSE_MATH__ 1
// AVX512F: #define __SSE__ 1
// AVX512F: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512cd -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512CD %s

// AVX512CD: #define __AVX2__ 1
// AVX512CD: #define __AVX512CD__ 1
// AVX512CD: #define __AVX512F__ 1
// AVX512CD: #define __AVX__ 1
// AVX512CD: #define __SSE2_MATH__ 1
// AVX512CD: #define __SSE2__ 1
// AVX512CD: #define __SSE3__ 1
// AVX512CD: #define __SSE4_1__ 1
// AVX512CD: #define __SSE4_2__ 1
// AVX512CD: #define __SSE_MATH__ 1
// AVX512CD: #define __SSE__ 1
// AVX512CD: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512er -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512ER %s

// AVX512ER: #define __AVX2__ 1
// AVX512ER: #define __AVX512ER__ 1
// AVX512ER: #define __AVX512F__ 1
// AVX512ER: #define __AVX__ 1
// AVX512ER: #define __SSE2_MATH__ 1
// AVX512ER: #define __SSE2__ 1
// AVX512ER: #define __SSE3__ 1
// AVX512ER: #define __SSE4_1__ 1
// AVX512ER: #define __SSE4_2__ 1
// AVX512ER: #define __SSE_MATH__ 1
// AVX512ER: #define __SSE__ 1
// AVX512ER: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512pf -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512PF %s

// AVX512PF: #define __AVX2__ 1
// AVX512PF: #define __AVX512F__ 1
// AVX512PF: #define __AVX512PF__ 1
// AVX512PF: #define __AVX__ 1
// AVX512PF: #define __SSE2_MATH__ 1
// AVX512PF: #define __SSE2__ 1
// AVX512PF: #define __SSE3__ 1
// AVX512PF: #define __SSE4_1__ 1
// AVX512PF: #define __SSE4_2__ 1
// AVX512PF: #define __SSE_MATH__ 1
// AVX512PF: #define __SSE__ 1
// AVX512PF: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512dq -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512DQ %s

// AVX512DQ: #define __AVX2__ 1
// AVX512DQ: #define __AVX512DQ__ 1
// AVX512DQ: #define __AVX512F__ 1
// AVX512DQ: #define __AVX__ 1
// AVX512DQ: #define __SSE2_MATH__ 1
// AVX512DQ: #define __SSE2__ 1
// AVX512DQ: #define __SSE3__ 1
// AVX512DQ: #define __SSE4_1__ 1
// AVX512DQ: #define __SSE4_2__ 1
// AVX512DQ: #define __SSE_MATH__ 1
// AVX512DQ: #define __SSE__ 1
// AVX512DQ: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512bw -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512BW %s

// AVX512BW: #define __AVX2__ 1
// AVX512BW: #define __AVX512BW__ 1
// AVX512BW: #define __AVX512F__ 1
// AVX512BW: #define __AVX__ 1
// AVX512BW: #define __SSE2_MATH__ 1
// AVX512BW: #define __SSE2__ 1
// AVX512BW: #define __SSE3__ 1
// AVX512BW: #define __SSE4_1__ 1
// AVX512BW: #define __SSE4_2__ 1
// AVX512BW: #define __SSE_MATH__ 1
// AVX512BW: #define __SSE__ 1
// AVX512BW: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512vl -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512VL %s

// AVX512VL: #define __AVX2__ 1
// AVX512VL: #define __AVX512F__ 1
// AVX512VL: #define __AVX512VL__ 1
// AVX512VL: #define __AVX__ 1
// AVX512VL: #define __SSE2_MATH__ 1
// AVX512VL: #define __SSE2__ 1
// AVX512VL: #define __SSE3__ 1
// AVX512VL: #define __SSE4_1__ 1
// AVX512VL: #define __SSE4_2__ 1
// AVX512VL: #define __SSE_MATH__ 1
// AVX512VL: #define __SSE__ 1
// AVX512VL: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512pf -mno-avx512f -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512F2 %s

// AVX512F2: #define __AVX2__ 1
// AVX512F2-NOT: #define __AVX512F__ 1
// AVX512F2-NOT: #define __AVX512PF__ 1
// AVX512F2: #define __AVX__ 1
// AVX512F2: #define __SSE2_MATH__ 1
// AVX512F2: #define __SSE2__ 1
// AVX512F2: #define __SSE3__ 1
// AVX512F2: #define __SSE4_1__ 1
// AVX512F2: #define __SSE4_2__ 1
// AVX512F2: #define __SSE_MATH__ 1
// AVX512F2: #define __SSE__ 1
// AVX512F2: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512ifma -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512IFMA %s

// AVX512IFMA: #define __AVX2__ 1
// AVX512IFMA: #define __AVX512F__ 1
// AVX512IFMA: #define __AVX512IFMA__ 1
// AVX512IFMA: #define __AVX__ 1
// AVX512IFMA: #define __SSE2_MATH__ 1
// AVX512IFMA: #define __SSE2__ 1
// AVX512IFMA: #define __SSE3__ 1
// AVX512IFMA: #define __SSE4_1__ 1
// AVX512IFMA: #define __SSE4_2__ 1
// AVX512IFMA: #define __SSE_MATH__ 1
// AVX512IFMA: #define __SSE__ 1
// AVX512IFMA: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512vbmi -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512VBMI %s

// AVX512VBMI: #define __AVX2__ 1
// AVX512VBMI: #define __AVX512BW__ 1
// AVX512VBMI: #define __AVX512F__ 1
// AVX512VBMI: #define __AVX512VBMI__ 1
// AVX512VBMI: #define __AVX__ 1
// AVX512VBMI: #define __SSE2_MATH__ 1
// AVX512VBMI: #define __SSE2__ 1
// AVX512VBMI: #define __SSE3__ 1
// AVX512VBMI: #define __SSE4_1__ 1
// AVX512VBMI: #define __SSE4_2__ 1
// AVX512VBMI: #define __SSE_MATH__ 1
// AVX512VBMI: #define __SSE__ 1
// AVX512VBMI: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512bitalg -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512BITALG %s

// AVX512BITALG: #define __AVX2__ 1
// AVX512BITALG: #define __AVX512BITALG__ 1
// AVX512BITALG: #define __AVX512BW__ 1
// AVX512BITALG: #define __AVX512F__ 1
// AVX512BITALG: #define __AVX__ 1
// AVX512BITALG: #define __SSE2_MATH__ 1
// AVX512BITALG: #define __SSE2__ 1
// AVX512BITALG: #define __SSE3__ 1
// AVX512BITALG: #define __SSE4_1__ 1
// AVX512BITALG: #define __SSE4_2__ 1
// AVX512BITALG: #define __SSE_MATH__ 1
// AVX512BITALG: #define __SSE__ 1
// AVX512BITALG: #define __SSSE3__ 1


// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512vbmi -mno-avx512bw -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512VBMINOAVX512BW %s

// AVX512VBMINOAVX512BW-NOT: #define __AVX512BW__ 1
// AVX512VBMINOAVX512BW-NOT: #define __AVX512VBMI__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512vbmi2 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512VBMI2 %s

// AVX512VBMI2: #define __AVX2__ 1
// AVX512VBMI2: #define __AVX512BW__ 1
// AVX512VBMI2: #define __AVX512F__ 1
// AVX512VBMI2: #define __AVX512VBMI2__ 1
// AVX512VBMI2: #define __AVX__ 1
// AVX512VBMI2: #define __SSE2_MATH__ 1
// AVX512VBMI2: #define __SSE2__ 1
// AVX512VBMI2: #define __SSE3__ 1
// AVX512VBMI2: #define __SSE4_1__ 1
// AVX512VBMI2: #define __SSE4_2__ 1
// AVX512VBMI2: #define __SSE_MATH__ 1
// AVX512VBMI2: #define __SSE__ 1
// AVX512VBMI2: #define __SSSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512vbmi2 -mno-avx512bw -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512VBMI2NOAVX512BW %s

// AVX512VBMI2NOAVX512BW-NOT: #define __AVX512BW__ 1
// AVX512VBMI2NOAVX512BW-NOT: #define __AVX512VBMI2__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512bitalg -mno-avx512bw -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512BITALGNOAVX512BW %s

// AVX512BITALGNOAVX512BW-NOT: #define __AVX512BITALG__ 1
// AVX512BITALGNOAVX512BW-NOT: #define __AVX512BW__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -msse4.2 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SSE42POPCNT %s

// SSE42POPCNT: #define __POPCNT__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mno-popcnt -msse4.2 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SSE42NOPOPCNT %s

// SSE42NOPOPCNT-NOT: #define __POPCNT__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mpopcnt -mno-sse4.2 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=NOSSE42POPCNT %s

// NOSSE42POPCNT: #define __POPCNT__ 1

// RUN: %clang -target i386-unknown-unknown -march=nehalem -mno-sse4.2 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=CPUPOPCNT %s
// RUN: %clang -target i386-unknown-unknown -march=silvermont -mno-sse4.2 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=CPUPOPCNT %s
// RUN: %clang -target i386-unknown-unknown -march=knl -mno-sse4.2 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=CPUPOPCNT %s

// CPUPOPCNT: #define __POPCNT__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentium -msse -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SSEMMX %s

// SSEMMX: #define __MMX__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentium -msse -mno-sse -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SSENOSSEMMX %s

// SSENOSSEMMX-NOT: #define __MMX__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentium -msse -mno-mmx -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SSENOMMX %s

// SSENOMMX-NOT: #define __MMX__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentium3 -mno-sse -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=MARCHMMXNOSSE %s
// RUN: %clang -target i386-unknown-unknown -march=atom -mno-sse -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=MARCHMMXNOSSE %s
// RUN: %clang -target i386-unknown-unknown -march=knl -mno-sse -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=MARCHMMXNOSSE %s
// RUN: %clang -target i386-unknown-unknown -march=btver1 -mno-sse -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=MARCHMMXNOSSE %s
// RUN: %clang -target i386-unknown-unknown -march=znver1 -mno-sse -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=MARCHMMXNOSSE %s

// MARCHMMXNOSSE: #define __MMX__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mf16c -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=F16C %s

// F16C: #define __AVX__ 1
// F16C: #define __F16C__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mf16c -mno-avx -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=F16CNOAVX %s

// F16CNOAVX-NOT: #define __AVX__ 1
// F16CNOAVX-NOT: #define __F16C__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentiumpro -mpclmul -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=PCLMUL %s

// PCLMUL: #define __PCLMUL__ 1
// PCLMUL: #define __SSE2__ 1
// PCLMUL-NOT: #define __SSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentiumpro -mpclmul -mno-sse2 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=PCLMULNOSSE2 %s

// PCLMULNOSSE2-NOT: #define __PCLMUL__ 1
// PCLMULNOSSE2-NOT: #define __SSE2__ 1
// PCLMULNOSSE2-NOT: #define __SSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentiumpro -maes -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AES %s

// AES: #define __AES__ 1
// AES: #define __SSE2__ 1
// AES-NOT: #define __SSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentiumpro -maes -mno-sse2 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AESNOSSE2 %s

// AESNOSSE2-NOT: #define __AES__ 1
// AESNOSSE2-NOT: #define __SSE2__ 1
// AESNOSSE2-NOT: #define __SSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentiumpro -mlwp -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=LWP %s

// LWP: #define __LWP__ 1

// RUN: %clang -target i386-unknown-unknown -march=bdver1 -mno-lwp -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=NOLWP %s

// NOLWP-NOT: #define __LWP__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentiumpro -msha -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SHA %s

// SHA: #define __SHA__ 1
// SHA: #define __SSE2__ 1
// SHA-NOT: #define __SSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentiumpro -msha -mno-sha -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SHANOSHA %s

// SHANOSHA-NOT: #define __SHA__ 1
// SHANOSHA-NOT: #define __SSE2__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentiumpro -msha -mno-sse2 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SHANOSSE2 %s

// SHANOSSE2-NOT: #define __SHA__ 1
// SHANOSSE2-NOT: #define __SSE2__ 1
// SHANOSSE2-NOT: #define __SSE3__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mtbm -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=TBM %s

// TBM: #define __TBM__ 1

// RUN: %clang -target i386-unknown-unknown -march=bdver2 -mno-tbm -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=NOTBM %s

// NOTBM-NOT: #define __TBM__ 1

// RUN: %clang -target i386-unknown-unknown -march=pentiumpro -mcx16 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=MCX16-32 %s

// MCX16-32-NOT: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16 1

// RUN: %clang -target x86_64-unknown-unknown -march=x86-64 -mcx16 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=MCX16-64 %s

// MCX16-64: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mprfchw -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=PRFCHW %s

// PRFCHW: #define __PRFCHW__ 1

// RUN: %clang -target i386-unknown-unknown -march=btver2 -mno-prfchw -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=NOPRFCHW %s

// NOPRFCHW-NOT: #define __PRFCHW__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -m3dnow -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=3DNOWPRFCHW %s

// 3DNOWPRFCHW: #define __3dNOW__ 1
// 3DNOWPRFCHW-NOT: #define __PRFCHW__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mno-prfchw -m3dnow -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=3DNOWNOPRFCHW %s

// 3DNOWNOPRFCHW: #define __3dNOW__ 1
// 3DNOWNOPRFCHW-NOT: #define __PRFCHW__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mprfchw -mno-3dnow -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=NO3DNOWPRFCHW %s

// NO3DNOWPRFCHW: #define __PRFCHW__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -madx -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=ADX %s

// ADX: #define __ADX__ 1

// RUN: %clang -target i386-unknown-unknown -mshstk -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SHSTK %s

// SHSTK: #define __SHSTK__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mrdseed -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=RDSEED %s

// RDSEED: #define __RDSEED__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mxsave -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=XSAVE %s

// XSAVE: #define __XSAVE__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mxsaveopt -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=XSAVEOPT %s

// XSAVEOPT: #define __XSAVEOPT__ 1
// XSAVEOPT: #define __XSAVE__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mxsavec -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=XSAVEC %s

// XSAVEC: #define __XSAVEC__ 1
// XSAVEC: #define __XSAVE__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mxsaves -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=XSAVES %s

// XSAVES: #define __XSAVES__ 1
// XSAVES: #define __XSAVE__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mxsaveopt -mxsavec -mxsaves -mno-xsave -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=NOXSAVE %s

// NOXSAVE-NOT: #define __XSAVEC__ 1
// NOXSAVE-NOT: #define __XSAVEOPT__ 1
// NOXSAVE-NOT: #define __XSAVES__ 1
// NOXSAVE-NOT: #define __XSAVE__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mclflushopt -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=CLFLUSHOPT %s

// CLFLUSHOPT: #define __CLFLUSHOPT__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mvaes -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=VAES %s

// VAES: #define __AES__ 1
// VAES: #define __VAES__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mvaes -mno-aes -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=VAESNOAES %s

// VAESNOAES-NOT: #define __AES__ 1
// VAESNOAES-NOT: #define __VAES__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mgfni -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=GFNI %s

// GFNI: #define __GFNI__ 1
// GFNI: #define __SSE2__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mvpclmulqdq -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=VPCLMULQDQ %s

// VPCLMULQDQ: #define __PCLMUL__ 1
// VPCLMULQDQ: #define __VPCLMULQDQ__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mvpclmulqdq -mno-pclmul -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=VPCLMULQDQNOPCLMUL %s
// VPCLMULQDQNOPCLMUL-NOT: #define __PCLMUL__ 1
// VPCLMULQDQNOPCLMUL-NOT: #define __VPCLMULQDQ__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mrdpid -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=RDPID %s

// RDPID: #define __RDPID__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512bf16 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512BF16 %s

// AVX512BF16: #define __AVX512BF16__ 1
// AVX512BF16: #define __AVX512BW__ 1
// AVX512BF16-NOT: #define __AVX512VL__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512bf16 -mno-avx512bw -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512BF16_NOAVX512BW %s

// AVX512BF16_NOAVX512BW-NOT: #define __AVX512BF16__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512bf16 -mno-avx512vl -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512BF16_NOAVX512VL %s

// AVX512BF16_NOAVX512VL: #define __AVX512BF16__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mavx512vp2intersect -x c -E -dM -o - %s | FileCheck  -check-prefix=VP2INTERSECT %s

// VP2INTERSECT: #define __AVX512F__ 1
// VP2INTERSECT: #define __AVX512VP2INTERSECT__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-avx512vp2intersect -x c -E -dM -o - %s | FileCheck  -check-prefix=NOVP2INTERSECT %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mavx512vp2intersect -mno-avx512f -x c -E -dM -o - %s | FileCheck  -check-prefix=NOVP2INTERSECT %s

// NOVP2INTERSECT-NOT: #define __AVX512VP2INTERSECT__ 1


// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mkl -x c -E -dM -o - %s | FileCheck  -check-prefix=KEYLOCKER %s
// KEYLOCKER: #define __KL__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-kl -x c -E -dM -o - %s | FileCheck  -check-prefix=NOKEYLOCKER %s
// NOKEYLOCKER-NOT: #define __KL__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mwidekl -x c -E -dM -o - %s | FileCheck  -check-prefix=KEYLOCKERW %s
// KEYLOCKERW: #define __KL__ 1
// KEYLOCKERW: #define __WIDEKL__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-widekl -x c -E -dM -o - %s | FileCheck  -check-prefix=NOKEYLOCKERW %s
// NOKEYLOCKERW-NOT: #define __KL__ 1
// NOKEYLOCKERW-NOT: #define __WIDEKL__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mwidekl -mno-kl -x c -E -dM -o - %s | FileCheck  -check-prefix=NOKEYLOCKERW2 %s
// NOKEYLOCKERW2-NOT: #define __KL__ 1
// NOKEYLOCKERW2-NOT: #define __WIDEKL__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -menqcmd -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=ENQCMD %s

// ENQCMD: #define __ENQCMD__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mno-enqcmd -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=NOENQCMD %s

// NOENQCMD-NOT: #define __ENQCMD__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mserialize -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=SERIALIZE %s

// SERIALIZE: #define __SERIALIZE__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mno-serialize -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=NOSERIALIZE %s

// NOSERIALIZE-NOT: #define __SERIALIZE__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mtsxldtrk -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=TSXLDTRK %s

// TSXLDTRK: #define __TSXLDTRK__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mno-tsxldtrk -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=NOTSXLDTRK %s

// NOTSXLDTRK-NOT: #define __TSXLDTRK__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mhreset -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=HRESET %s

// HRESET: #define __HRESET__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mno-hreset -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=NOHRESET %s

// NOHRESET-NOT: #define __HRESET__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -muintr -x c -E -dM -o - %s | FileCheck -check-prefix=UINTR %s

// UINTR: #define __UINTR__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-uintr -x c -E -dM -o - %s | FileCheck -check-prefix=NOUINTR %s

// NOUINTR-NOT: #define __UINTR__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavxvnni -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVXVNNI %s

// AVXVNNI: #define __AVX2__ 1
// AVXVNNI: #define __AVXVNNI__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mno-avxvnni -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=NOAVXVNNI %s

// NOAVXVNNI-NOT: #define __AVXVNNI__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavxvnni -mno-avx2 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVXVNNINOAVX2 %s

// AVXVNNINOAVX2-NOT: #define __AVX2__ 1
// AVXVNNINOAVX2-NOT: #define __AVXVNNI__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512fp16 -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512FP16 %s

// AVX512FP16: #define __AVX512BW__ 1
// AVX512FP16: #define __AVX512DQ__ 1
// AVX512FP16: #define __AVX512FP16__ 1
// AVX512FP16: #define __AVX512VL__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512fp16 -mno-avx512vl -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512FP16NOAVX512VL %s

// AVX512FP16NOAVX512VL-NOT: #define __AVX512FP16__ 1
// AVX512FP16NOAVX512VL-NOT: #define __AVX512VL__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512fp16 -mno-avx512bw -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512FP16NOAVX512BW %s

// AVX512FP16NOAVX512BW-NOT: #define __AVX512BW__ 1
// AVX512FP16NOAVX512BW-NOT: #define __AVX512FP16__ 1

// RUN: %clang -target i386-unknown-unknown -march=atom -mavx512fp16 -mno-avx512dq -x c -E -dM -o - %s | FileCheck -match-full-lines --check-prefix=AVX512FP16NOAVX512DQ %s

// AVX512FP16NOAVX512DQ-NOT: #define __AVX512DQ__ 1
// AVX512FP16NOAVX512DQ-NOT: #define __AVX512FP16__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mcrc32 -x c -E -dM -o - %s | FileCheck -check-prefix=CRC32 %s

// CRC32: #define __CRC32__ 1

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-crc32 -x c -E -dM -o - %s | FileCheck -check-prefix=NOCRC32 %s

// NOCRC32-NOT: #define __CRC32__ 1
