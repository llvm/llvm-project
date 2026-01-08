// RUN: %clang -target x86_64 -march=x86-64 -E -dM %s > %tv1
// RUN: FileCheck %s --check-prefix=X86_64_V1 < %tv1

// X86_64_V1: #define __MMX__ 1
// X86_64_V1: #define __SSE2_MATH__ 1
// X86_64_V1: #define __SSE2__ 1
// X86_64_V1: #define __SSE_MATH__ 1
// X86_64_V1: #define __SSE__ 1
// X86_64_V1: #define __amd64 1
// X86_64_V1: #define __amd64__ 1
// X86_64_V1: #define __k8 1
// X86_64_V1: #define __k8__ 1
// X86_64_V1: #define __x86_64 1
// X86_64_V1: #define __x86_64__ 1

// RUN: %clang -target x86_64 -march=x86-64-v2 -E -dM %s > %tv2
// RUN: diff %tv1 %tv2 > %t.txt || true
// RUN: FileCheck %s --check-prefix=X86_64_V2 < %t.txt

/// v2 is close to Nehalem.
// X86_64_V2:      #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_16 1
// X86_64_V2:      #define __LAHF_SAHF__ 1
// X86_64_V2:      #define __POPCNT__ 1
// X86_64_V2:      #define __SSE3__ 1
// X86_64_V2-NEXT: #define __SSE4_1__ 1
// X86_64_V2-NEXT: #define __SSE4_2__ 1
// X86_64_V2:      #define __SSSE3__ 1

/// v3 is close to Haswell.
// RUN: %clang -target x86_64 -march=x86-64-v3 -E -dM %s > %tv3
// RUN: diff %tv2 %tv3 > %t.txt || true
// RUN: FileCheck %s --check-prefix=X86_64_V3 < %t.txt

// X86_64_V3:      #define __AVX2__ 1
// X86_64_V3-NEXT: #define __AVX__ 1
// X86_64_V3:      #define __BMI2__ 1
// X86_64_V3-NEXT: #define __BMI__ 1
// X86_64_V3:      #define __F16C__ 1
// X86_64_V3:      #define __FMA__ 1
// X86_64_V3:      #define __LZCNT__ 1
// X86_64_V3:      #define __MOVBE__ 1
// X86_64_V3:      #define __XSAVE__ 1

/// v4 is close to the AVX-512 level implemented by Xeon Scalable Processors.
// RUN: %clang -target x86_64 -march=x86-64-v4 -E -dM %s > %tv4
// RUN: diff %tv3 %tv4 > %t.txt || true
// RUN: FileCheck %s --check-prefix=X86_64_V4 < %t.txt

// X86_64_V4:      #define __AVX512BW__ 1
// X86_64_V4-NEXT: #define __AVX512CD__ 1
// X86_64_V4-NEXT: #define __AVX512DQ__ 1
// X86_64_V4-NEXT: #define __AVX512F__ 1
// X86_64_V4-NEXT: #define __AVX512VL__ 1
// X86_64_V4-NOT:  #define __AVX512{{.*}}
