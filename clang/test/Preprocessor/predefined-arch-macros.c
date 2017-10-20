// Begin X86/GCC/Linux tests ----------------
//
// RUN: %clang -march=i386 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_I386_M32
// CHECK_I386_M32: #define __i386 1
// CHECK_I386_M32: #define __i386__ 1
// CHECK_I386_M32: #define __tune_i386__ 1
// CHECK_I386_M32: #define i386 1
// RUN: not %clang -march=i386 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_I386_M64
// CHECK_I386_M64: error: {{.*}}
//
// RUN: %clang -march=i486 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_I486_M32
// CHECK_I486_M32: #define __i386 1
// CHECK_I486_M32: #define __i386__ 1
// CHECK_I486_M32: #define __i486 1
// CHECK_I486_M32: #define __i486__ 1
// CHECK_I486_M32: #define __tune_i486__ 1
// CHECK_I486_M32: #define i386 1
// RUN: not %clang -march=i486 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_I486_M64
// CHECK_I486_M64: error: {{.*}}
//
// RUN: %clang -march=i586 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_I586_M32
// CHECK_I586_M32: #define __i386 1
// CHECK_I586_M32: #define __i386__ 1
// CHECK_I586_M32: #define __i586 1
// CHECK_I586_M32: #define __i586__ 1
// CHECK_I586_M32: #define __pentium 1
// CHECK_I586_M32: #define __pentium__ 1
// CHECK_I586_M32: #define __tune_i586__ 1
// CHECK_I586_M32: #define __tune_pentium__ 1
// CHECK_I586_M32: #define i386 1
// RUN: not %clang -march=i586 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_I586_M64
// CHECK_I586_M64: error: {{.*}}
//
// RUN: %clang -march=pentium -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM_M32
// CHECK_PENTIUM_M32: #define __i386 1
// CHECK_PENTIUM_M32: #define __i386__ 1
// CHECK_PENTIUM_M32: #define __i586 1
// CHECK_PENTIUM_M32: #define __i586__ 1
// CHECK_PENTIUM_M32: #define __pentium 1
// CHECK_PENTIUM_M32: #define __pentium__ 1
// CHECK_PENTIUM_M32: #define __tune_i586__ 1
// CHECK_PENTIUM_M32: #define __tune_pentium__ 1
// CHECK_PENTIUM_M32: #define i386 1
// RUN: not %clang -march=pentium -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM_M64
// CHECK_PENTIUM_M64: error: {{.*}}
//
// RUN: %clang -march=pentium-mmx -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM_MMX_M32
// CHECK_PENTIUM_MMX_M32: #define __MMX__ 1
// CHECK_PENTIUM_MMX_M32: #define __i386 1
// CHECK_PENTIUM_MMX_M32: #define __i386__ 1
// CHECK_PENTIUM_MMX_M32: #define __i586 1
// CHECK_PENTIUM_MMX_M32: #define __i586__ 1
// CHECK_PENTIUM_MMX_M32: #define __pentium 1
// CHECK_PENTIUM_MMX_M32: #define __pentium__ 1
// CHECK_PENTIUM_MMX_M32: #define __pentium_mmx__ 1
// CHECK_PENTIUM_MMX_M32: #define __tune_i586__ 1
// CHECK_PENTIUM_MMX_M32: #define __tune_pentium__ 1
// CHECK_PENTIUM_MMX_M32: #define __tune_pentium_mmx__ 1
// CHECK_PENTIUM_MMX_M32: #define i386 1
// RUN: not %clang -march=pentium-mmx -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM_MMX_M64
// CHECK_PENTIUM_MMX_M64: error: {{.*}}
//
// RUN: %clang -march=winchip-c6 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_WINCHIP_C6_M32
// CHECK_WINCHIP_C6_M32: #define __MMX__ 1
// CHECK_WINCHIP_C6_M32: #define __i386 1
// CHECK_WINCHIP_C6_M32: #define __i386__ 1
// CHECK_WINCHIP_C6_M32: #define __i486 1
// CHECK_WINCHIP_C6_M32: #define __i486__ 1
// CHECK_WINCHIP_C6_M32: #define __tune_i486__ 1
// CHECK_WINCHIP_C6_M32: #define i386 1
// RUN: not %clang -march=winchip-c6 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_WINCHIP_C6_M64
// CHECK_WINCHIP_C6_M64: error: {{.*}}
//
// RUN: %clang -march=winchip2 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_WINCHIP2_M32
// CHECK_WINCHIP2_M32: #define __3dNOW__ 1
// CHECK_WINCHIP2_M32: #define __MMX__ 1
// CHECK_WINCHIP2_M32: #define __i386 1
// CHECK_WINCHIP2_M32: #define __i386__ 1
// CHECK_WINCHIP2_M32: #define __i486 1
// CHECK_WINCHIP2_M32: #define __i486__ 1
// CHECK_WINCHIP2_M32: #define __tune_i486__ 1
// CHECK_WINCHIP2_M32: #define i386 1
// RUN: not %clang -march=winchip2 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_WINCHIP2_M64
// CHECK_WINCHIP2_M64: error: {{.*}}
//
// RUN: %clang -march=c3 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_C3_M32
// CHECK_C3_M32: #define __3dNOW__ 1
// CHECK_C3_M32: #define __MMX__ 1
// CHECK_C3_M32: #define __i386 1
// CHECK_C3_M32: #define __i386__ 1
// CHECK_C3_M32: #define __i486 1
// CHECK_C3_M32: #define __i486__ 1
// CHECK_C3_M32: #define __tune_i486__ 1
// CHECK_C3_M32: #define i386 1
// RUN: not %clang -march=c3 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_C3_M64
// CHECK_C3_M64: error: {{.*}}
//
// RUN: %clang -march=c3-2 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_C3_2_M32
// CHECK_C3_2_M32: #define __MMX__ 1
// CHECK_C3_2_M32: #define __SSE__ 1
// CHECK_C3_2_M32: #define __i386 1
// CHECK_C3_2_M32: #define __i386__ 1
// CHECK_C3_2_M32: #define __i686 1
// CHECK_C3_2_M32: #define __i686__ 1
// CHECK_C3_2_M32: #define __pentiumpro 1
// CHECK_C3_2_M32: #define __pentiumpro__ 1
// CHECK_C3_2_M32: #define __tune_i686__ 1
// CHECK_C3_2_M32: #define __tune_pentium2__ 1
// CHECK_C3_2_M32: #define __tune_pentiumpro__ 1
// CHECK_C3_2_M32: #define i386 1
// RUN: not %clang -march=c3-2 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_C3_2_M64
// CHECK_C3_2_M64: error: {{.*}}
//
// RUN: %clang -march=i686 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_I686_M32
// CHECK_I686_M32: #define __i386 1
// CHECK_I686_M32: #define __i386__ 1
// CHECK_I686_M32: #define __i686 1
// CHECK_I686_M32: #define __i686__ 1
// CHECK_I686_M32: #define __pentiumpro 1
// CHECK_I686_M32: #define __pentiumpro__ 1
// CHECK_I686_M32: #define i386 1
// RUN: not %clang -march=i686 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_I686_M64
// CHECK_I686_M64: error: {{.*}}
//
// RUN: %clang -march=pentiumpro -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUMPRO_M32
// CHECK_PENTIUMPRO_M32: #define __i386 1
// CHECK_PENTIUMPRO_M32: #define __i386__ 1
// CHECK_PENTIUMPRO_M32: #define __i686 1
// CHECK_PENTIUMPRO_M32: #define __i686__ 1
// CHECK_PENTIUMPRO_M32: #define __pentiumpro 1
// CHECK_PENTIUMPRO_M32: #define __pentiumpro__ 1
// CHECK_PENTIUMPRO_M32: #define __tune_i686__ 1
// CHECK_PENTIUMPRO_M32: #define __tune_pentiumpro__ 1
// CHECK_PENTIUMPRO_M32: #define i386 1
// RUN: not %clang -march=pentiumpro -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUMPRO_M64
// CHECK_PENTIUMPRO_M64: error: {{.*}}
//
// RUN: %clang -march=pentium2 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM2_M32
// CHECK_PENTIUM2_M32: #define __MMX__ 1
// CHECK_PENTIUM2_M32: #define __i386 1
// CHECK_PENTIUM2_M32: #define __i386__ 1
// CHECK_PENTIUM2_M32: #define __i686 1
// CHECK_PENTIUM2_M32: #define __i686__ 1
// CHECK_PENTIUM2_M32: #define __pentiumpro 1
// CHECK_PENTIUM2_M32: #define __pentiumpro__ 1
// CHECK_PENTIUM2_M32: #define __tune_i686__ 1
// CHECK_PENTIUM2_M32: #define __tune_pentium2__ 1
// CHECK_PENTIUM2_M32: #define __tune_pentiumpro__ 1
// CHECK_PENTIUM2_M32: #define i386 1
// RUN: not %clang -march=pentium2 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM2_M64
// CHECK_PENTIUM2_M64: error: {{.*}}
//
// RUN: %clang -march=pentium3 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM3_M32
// CHECK_PENTIUM3_M32: #define __MMX__ 1
// CHECK_PENTIUM3_M32: #define __SSE__ 1
// CHECK_PENTIUM3_M32: #define __i386 1
// CHECK_PENTIUM3_M32: #define __i386__ 1
// CHECK_PENTIUM3_M32: #define __i686 1
// CHECK_PENTIUM3_M32: #define __i686__ 1
// CHECK_PENTIUM3_M32: #define __pentiumpro 1
// CHECK_PENTIUM3_M32: #define __pentiumpro__ 1
// CHECK_PENTIUM3_M32: #define __tune_i686__ 1
// CHECK_PENTIUM3_M32: #define __tune_pentium2__ 1
// CHECK_PENTIUM3_M32: #define __tune_pentium3__ 1
// CHECK_PENTIUM3_M32: #define __tune_pentiumpro__ 1
// CHECK_PENTIUM3_M32: #define i386 1
// RUN: not %clang -march=pentium3 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM3_M64
// CHECK_PENTIUM3_M64: error: {{.*}}
//
// RUN: %clang -march=pentium3m -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM3M_M32
// CHECK_PENTIUM3M_M32: #define __MMX__ 1
// CHECK_PENTIUM3M_M32: #define __SSE__ 1
// CHECK_PENTIUM3M_M32: #define __i386 1
// CHECK_PENTIUM3M_M32: #define __i386__ 1
// CHECK_PENTIUM3M_M32: #define __i686 1
// CHECK_PENTIUM3M_M32: #define __i686__ 1
// CHECK_PENTIUM3M_M32: #define __pentiumpro 1
// CHECK_PENTIUM3M_M32: #define __pentiumpro__ 1
// CHECK_PENTIUM3M_M32: #define __tune_i686__ 1
// CHECK_PENTIUM3M_M32: #define __tune_pentiumpro__ 1
// CHECK_PENTIUM3M_M32: #define i386 1
// RUN: not %clang -march=pentium3m -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM3M_M64
// CHECK_PENTIUM3M_M64: error: {{.*}}
//
// RUN: %clang -march=pentium-m -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM_M_M32
// CHECK_PENTIUM_M_M32: #define __MMX__ 1
// CHECK_PENTIUM_M_M32: #define __SSE2__ 1
// CHECK_PENTIUM_M_M32: #define __SSE__ 1
// CHECK_PENTIUM_M_M32: #define __i386 1
// CHECK_PENTIUM_M_M32: #define __i386__ 1
// CHECK_PENTIUM_M_M32: #define __i686 1
// CHECK_PENTIUM_M_M32: #define __i686__ 1
// CHECK_PENTIUM_M_M32: #define __pentiumpro 1
// CHECK_PENTIUM_M_M32: #define __pentiumpro__ 1
// CHECK_PENTIUM_M_M32: #define __tune_i686__ 1
// CHECK_PENTIUM_M_M32: #define __tune_pentiumpro__ 1
// CHECK_PENTIUM_M_M32: #define i386 1
// RUN: not %clang -march=pentium-m -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM_M_M64
// CHECK_PENTIUM_M_M64: error: {{.*}}
//
// RUN: %clang -march=pentium4 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM4_M32
// CHECK_PENTIUM4_M32: #define __MMX__ 1
// CHECK_PENTIUM4_M32: #define __SSE2__ 1
// CHECK_PENTIUM4_M32: #define __SSE__ 1
// CHECK_PENTIUM4_M32: #define __i386 1
// CHECK_PENTIUM4_M32: #define __i386__ 1
// CHECK_PENTIUM4_M32: #define __pentium4 1
// CHECK_PENTIUM4_M32: #define __pentium4__ 1
// CHECK_PENTIUM4_M32: #define __tune_pentium4__ 1
// CHECK_PENTIUM4_M32: #define i386 1
// RUN: not %clang -march=pentium4 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM4_M64
// CHECK_PENTIUM4_M64: error: {{.*}}
//
// RUN: %clang -march=pentium4m -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM4M_M32
// CHECK_PENTIUM4M_M32: #define __MMX__ 1
// CHECK_PENTIUM4M_M32: #define __SSE2__ 1
// CHECK_PENTIUM4M_M32: #define __SSE__ 1
// CHECK_PENTIUM4M_M32: #define __i386 1
// CHECK_PENTIUM4M_M32: #define __i386__ 1
// CHECK_PENTIUM4M_M32: #define __pentium4 1
// CHECK_PENTIUM4M_M32: #define __pentium4__ 1
// CHECK_PENTIUM4M_M32: #define __tune_pentium4__ 1
// CHECK_PENTIUM4M_M32: #define i386 1
// RUN: not %clang -march=pentium4m -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PENTIUM4M_M64
// CHECK_PENTIUM4M_M64: error: {{.*}}
//
// RUN: %clang -march=prescott -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PRESCOTT_M32
// CHECK_PRESCOTT_M32: #define __MMX__ 1
// CHECK_PRESCOTT_M32: #define __SSE2__ 1
// CHECK_PRESCOTT_M32: #define __SSE3__ 1
// CHECK_PRESCOTT_M32: #define __SSE__ 1
// CHECK_PRESCOTT_M32: #define __i386 1
// CHECK_PRESCOTT_M32: #define __i386__ 1
// CHECK_PRESCOTT_M32: #define __nocona 1
// CHECK_PRESCOTT_M32: #define __nocona__ 1
// CHECK_PRESCOTT_M32: #define __tune_nocona__ 1
// CHECK_PRESCOTT_M32: #define i386 1
// RUN: not %clang -march=prescott -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PRESCOTT_M64
// CHECK_PRESCOTT_M64: error: {{.*}}
//
// RUN: %clang -march=nocona -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_NOCONA_M32
// CHECK_NOCONA_M32: #define __MMX__ 1
// CHECK_NOCONA_M32: #define __SSE2__ 1
// CHECK_NOCONA_M32: #define __SSE3__ 1
// CHECK_NOCONA_M32: #define __SSE__ 1
// CHECK_NOCONA_M32: #define __i386 1
// CHECK_NOCONA_M32: #define __i386__ 1
// CHECK_NOCONA_M32: #define __nocona 1
// CHECK_NOCONA_M32: #define __nocona__ 1
// CHECK_NOCONA_M32: #define __tune_nocona__ 1
// CHECK_NOCONA_M32: #define i386 1
// RUN: %clang -march=nocona -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_NOCONA_M64
// CHECK_NOCONA_M64: #define __MMX__ 1
// CHECK_NOCONA_M64: #define __SSE2_MATH__ 1
// CHECK_NOCONA_M64: #define __SSE2__ 1
// CHECK_NOCONA_M64: #define __SSE3__ 1
// CHECK_NOCONA_M64: #define __SSE_MATH__ 1
// CHECK_NOCONA_M64: #define __SSE__ 1
// CHECK_NOCONA_M64: #define __amd64 1
// CHECK_NOCONA_M64: #define __amd64__ 1
// CHECK_NOCONA_M64: #define __nocona 1
// CHECK_NOCONA_M64: #define __nocona__ 1
// CHECK_NOCONA_M64: #define __tune_nocona__ 1
// CHECK_NOCONA_M64: #define __x86_64 1
// CHECK_NOCONA_M64: #define __x86_64__ 1
//
// RUN: %clang -march=core2 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_CORE2_M32
// CHECK_CORE2_M32: #define __MMX__ 1
// CHECK_CORE2_M32: #define __SSE2__ 1
// CHECK_CORE2_M32: #define __SSE3__ 1
// CHECK_CORE2_M32: #define __SSE__ 1
// CHECK_CORE2_M32: #define __SSSE3__ 1
// CHECK_CORE2_M32: #define __core2 1
// CHECK_CORE2_M32: #define __core2__ 1
// CHECK_CORE2_M32: #define __i386 1
// CHECK_CORE2_M32: #define __i386__ 1
// CHECK_CORE2_M32: #define __tune_core2__ 1
// CHECK_CORE2_M32: #define i386 1
// RUN: %clang -march=core2 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_CORE2_M64
// CHECK_CORE2_M64: #define __MMX__ 1
// CHECK_CORE2_M64: #define __SSE2_MATH__ 1
// CHECK_CORE2_M64: #define __SSE2__ 1
// CHECK_CORE2_M64: #define __SSE3__ 1
// CHECK_CORE2_M64: #define __SSE_MATH__ 1
// CHECK_CORE2_M64: #define __SSE__ 1
// CHECK_CORE2_M64: #define __SSSE3__ 1
// CHECK_CORE2_M64: #define __amd64 1
// CHECK_CORE2_M64: #define __amd64__ 1
// CHECK_CORE2_M64: #define __core2 1
// CHECK_CORE2_M64: #define __core2__ 1
// CHECK_CORE2_M64: #define __tune_core2__ 1
// CHECK_CORE2_M64: #define __x86_64 1
// CHECK_CORE2_M64: #define __x86_64__ 1
//
// RUN: %clang -march=corei7 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_COREI7_M32
// CHECK_COREI7_M32: #define __MMX__ 1
// CHECK_COREI7_M32: #define __POPCNT__ 1
// CHECK_COREI7_M32: #define __SSE2__ 1
// CHECK_COREI7_M32: #define __SSE3__ 1
// CHECK_COREI7_M32: #define __SSE4_1__ 1
// CHECK_COREI7_M32: #define __SSE4_2__ 1
// CHECK_COREI7_M32: #define __SSE__ 1
// CHECK_COREI7_M32: #define __SSSE3__ 1
// CHECK_COREI7_M32: #define __corei7 1
// CHECK_COREI7_M32: #define __corei7__ 1
// CHECK_COREI7_M32: #define __i386 1
// CHECK_COREI7_M32: #define __i386__ 1
// CHECK_COREI7_M32: #define __tune_corei7__ 1
// CHECK_COREI7_M32: #define i386 1
// RUN: %clang -march=corei7 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_COREI7_M64
// CHECK_COREI7_M64: #define __MMX__ 1
// CHECK_COREI7_M64: #define __POPCNT__ 1
// CHECK_COREI7_M64: #define __SSE2_MATH__ 1
// CHECK_COREI7_M64: #define __SSE2__ 1
// CHECK_COREI7_M64: #define __SSE3__ 1
// CHECK_COREI7_M64: #define __SSE4_1__ 1
// CHECK_COREI7_M64: #define __SSE4_2__ 1
// CHECK_COREI7_M64: #define __SSE_MATH__ 1
// CHECK_COREI7_M64: #define __SSE__ 1
// CHECK_COREI7_M64: #define __SSSE3__ 1
// CHECK_COREI7_M64: #define __amd64 1
// CHECK_COREI7_M64: #define __amd64__ 1
// CHECK_COREI7_M64: #define __corei7 1
// CHECK_COREI7_M64: #define __corei7__ 1
// CHECK_COREI7_M64: #define __tune_corei7__ 1
// CHECK_COREI7_M64: #define __x86_64 1
// CHECK_COREI7_M64: #define __x86_64__ 1
//
// RUN: %clang -march=corei7-avx -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_COREI7_AVX_M32
// CHECK_COREI7_AVX_M32: #define __AES__ 1
// CHECK_COREI7_AVX_M32: #define __AVX__ 1
// CHECK_COREI7_AVX_M32: #define __MMX__ 1
// CHECK_COREI7_AVX_M32: #define __PCLMUL__ 1
// CHECK_COREI7_AVX_M32-NOT: __RDRND__
// CHECK_COREI7_AVX_M32: #define __POPCNT__ 1
// CHECK_COREI7_AVX_M32: #define __SSE2__ 1
// CHECK_COREI7_AVX_M32: #define __SSE3__ 1
// CHECK_COREI7_AVX_M32: #define __SSE4_1__ 1
// CHECK_COREI7_AVX_M32: #define __SSE4_2__ 1
// CHECK_COREI7_AVX_M32: #define __SSE__ 1
// CHECK_COREI7_AVX_M32: #define __SSSE3__ 1
// CHECK_COREI7_AVX_M32: #define __XSAVEOPT__ 1
// CHECK_COREI7_AVX_M32: #define __XSAVE__ 1
// CHECK_COREI7_AVX_M32: #define __corei7 1
// CHECK_COREI7_AVX_M32: #define __corei7__ 1
// CHECK_COREI7_AVX_M32: #define __i386 1
// CHECK_COREI7_AVX_M32: #define __i386__ 1
// CHECK_COREI7_AVX_M32: #define __tune_corei7__ 1
// CHECK_COREI7_AVX_M32: #define i386 1
// RUN: %clang -march=corei7-avx -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_COREI7_AVX_M64
// CHECK_COREI7_AVX_M64: #define __AES__ 1
// CHECK_COREI7_AVX_M64: #define __AVX__ 1
// CHECK_COREI7_AVX_M64: #define __MMX__ 1
// CHECK_COREI7_AVX_M64: #define __PCLMUL__ 1
// CHECK_COREI7_AVX_M64-NOT: __RDRND__
// CHECK_COREI7_AVX_M64: #define __POPCNT__ 1
// CHECK_COREI7_AVX_M64: #define __SSE2_MATH__ 1
// CHECK_COREI7_AVX_M64: #define __SSE2__ 1
// CHECK_COREI7_AVX_M64: #define __SSE3__ 1
// CHECK_COREI7_AVX_M64: #define __SSE4_1__ 1
// CHECK_COREI7_AVX_M64: #define __SSE4_2__ 1
// CHECK_COREI7_AVX_M64: #define __SSE_MATH__ 1
// CHECK_COREI7_AVX_M64: #define __SSE__ 1
// CHECK_COREI7_AVX_M64: #define __SSSE3__ 1
// CHECK_COREI7_AVX_M64: #define __XSAVEOPT__ 1
// CHECK_COREI7_AVX_M64: #define __XSAVE__ 1
// CHECK_COREI7_AVX_M64: #define __amd64 1
// CHECK_COREI7_AVX_M64: #define __amd64__ 1
// CHECK_COREI7_AVX_M64: #define __corei7 1
// CHECK_COREI7_AVX_M64: #define __corei7__ 1
// CHECK_COREI7_AVX_M64: #define __tune_corei7__ 1
// CHECK_COREI7_AVX_M64: #define __x86_64 1
// CHECK_COREI7_AVX_M64: #define __x86_64__ 1
//
// RUN: %clang -march=core-avx-i -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_CORE_AVX_I_M32
// CHECK_CORE_AVX_I_M32: #define __AES__ 1
// CHECK_CORE_AVX_I_M32: #define __AVX__ 1
// CHECK_CORE_AVX_I_M32: #define __F16C__ 1
// CHECK_CORE_AVX_I_M32: #define __MMX__ 1
// CHECK_CORE_AVX_I_M32: #define __PCLMUL__ 1
// CHECK_CORE_AVX_I_M32: #define __RDRND__ 1
// CHECK_CORE_AVX_I_M32: #define __SSE2__ 1
// CHECK_CORE_AVX_I_M32: #define __SSE3__ 1
// CHECK_CORE_AVX_I_M32: #define __SSE4_1__ 1
// CHECK_CORE_AVX_I_M32: #define __SSE4_2__ 1
// CHECK_CORE_AVX_I_M32: #define __SSE__ 1
// CHECK_CORE_AVX_I_M32: #define __SSSE3__ 1
// CHECK_CORE_AVX_I_M32: #define __XSAVEOPT__ 1
// CHECK_CORE_AVX_I_M32: #define __XSAVE__ 1
// CHECK_CORE_AVX_I_M32: #define __corei7 1
// CHECK_CORE_AVX_I_M32: #define __corei7__ 1
// CHECK_CORE_AVX_I_M32: #define __i386 1
// CHECK_CORE_AVX_I_M32: #define __i386__ 1
// CHECK_CORE_AVX_I_M32: #define __tune_corei7__ 1
// CHECK_CORE_AVX_I_M32: #define i386 1
// RUN: %clang -march=core-avx-i -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_CORE_AVX_I_M64
// CHECK_CORE_AVX_I_M64: #define __AES__ 1
// CHECK_CORE_AVX_I_M64: #define __AVX__ 1
// CHECK_CORE_AVX_I_M64: #define __F16C__ 1
// CHECK_CORE_AVX_I_M64: #define __MMX__ 1
// CHECK_CORE_AVX_I_M64: #define __PCLMUL__ 1
// CHECK_CORE_AVX_I_M64: #define __RDRND__ 1
// CHECK_CORE_AVX_I_M64: #define __SSE2_MATH__ 1
// CHECK_CORE_AVX_I_M64: #define __SSE2__ 1
// CHECK_CORE_AVX_I_M64: #define __SSE3__ 1
// CHECK_CORE_AVX_I_M64: #define __SSE4_1__ 1
// CHECK_CORE_AVX_I_M64: #define __SSE4_2__ 1
// CHECK_CORE_AVX_I_M64: #define __SSE_MATH__ 1
// CHECK_CORE_AVX_I_M64: #define __SSE__ 1
// CHECK_CORE_AVX_I_M64: #define __SSSE3__ 1
// CHECK_CORE_AVX_I_M64: #define __XSAVEOPT__ 1
// CHECK_CORE_AVX_I_M64: #define __XSAVE__ 1
// CHECK_CORE_AVX_I_M64: #define __amd64 1
// CHECK_CORE_AVX_I_M64: #define __amd64__ 1
// CHECK_CORE_AVX_I_M64: #define __corei7 1
// CHECK_CORE_AVX_I_M64: #define __corei7__ 1
// CHECK_CORE_AVX_I_M64: #define __tune_corei7__ 1
// CHECK_CORE_AVX_I_M64: #define __x86_64 1
// CHECK_CORE_AVX_I_M64: #define __x86_64__ 1
//
// RUN: %clang -march=core-avx2 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_CORE_AVX2_M32
// CHECK_CORE_AVX2_M32: #define __AES__ 1
// CHECK_CORE_AVX2_M32: #define __AVX2__ 1
// CHECK_CORE_AVX2_M32: #define __AVX__ 1
// CHECK_CORE_AVX2_M32: #define __BMI2__ 1
// CHECK_CORE_AVX2_M32: #define __BMI__ 1
// CHECK_CORE_AVX2_M32: #define __F16C__ 1
// CHECK_CORE_AVX2_M32: #define __FMA__ 1
// CHECK_CORE_AVX2_M32: #define __LZCNT__ 1
// CHECK_CORE_AVX2_M32: #define __MMX__ 1
// CHECK_CORE_AVX2_M32: #define __PCLMUL__ 1
// CHECK_CORE_AVX2_M32: #define __POPCNT__ 1
// CHECK_CORE_AVX2_M32: #define __RDRND__ 1
// CHECK_CORE_AVX2_M32: #define __SSE2__ 1
// CHECK_CORE_AVX2_M32: #define __SSE3__ 1
// CHECK_CORE_AVX2_M32: #define __SSE4_1__ 1
// CHECK_CORE_AVX2_M32: #define __SSE4_2__ 1
// CHECK_CORE_AVX2_M32: #define __SSE__ 1
// CHECK_CORE_AVX2_M32: #define __SSSE3__ 1
// CHECK_CORE_AVX2_M32: #define __XSAVEOPT__ 1
// CHECK_CORE_AVX2_M32: #define __XSAVE__ 1
// CHECK_CORE_AVX2_M32: #define __corei7 1
// CHECK_CORE_AVX2_M32: #define __corei7__ 1
// CHECK_CORE_AVX2_M32: #define __i386 1
// CHECK_CORE_AVX2_M32: #define __i386__ 1
// CHECK_CORE_AVX2_M32: #define __tune_corei7__ 1
// CHECK_CORE_AVX2_M32: #define i386 1
// RUN: %clang -march=core-avx2 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_CORE_AVX2_M64
// CHECK_CORE_AVX2_M64: #define __AES__ 1
// CHECK_CORE_AVX2_M64: #define __AVX2__ 1
// CHECK_CORE_AVX2_M64: #define __AVX__ 1
// CHECK_CORE_AVX2_M64: #define __BMI2__ 1
// CHECK_CORE_AVX2_M64: #define __BMI__ 1
// CHECK_CORE_AVX2_M64: #define __F16C__ 1
// CHECK_CORE_AVX2_M64: #define __FMA__ 1
// CHECK_CORE_AVX2_M64: #define __LZCNT__ 1
// CHECK_CORE_AVX2_M64: #define __MMX__ 1
// CHECK_CORE_AVX2_M64: #define __PCLMUL__ 1
// CHECK_CORE_AVX2_M64: #define __POPCNT__ 1
// CHECK_CORE_AVX2_M64: #define __RDRND__ 1
// CHECK_CORE_AVX2_M64: #define __SSE2_MATH__ 1
// CHECK_CORE_AVX2_M64: #define __SSE2__ 1
// CHECK_CORE_AVX2_M64: #define __SSE3__ 1
// CHECK_CORE_AVX2_M64: #define __SSE4_1__ 1
// CHECK_CORE_AVX2_M64: #define __SSE4_2__ 1
// CHECK_CORE_AVX2_M64: #define __SSE_MATH__ 1
// CHECK_CORE_AVX2_M64: #define __SSE__ 1
// CHECK_CORE_AVX2_M64: #define __SSSE3__ 1
// CHECK_CORE_AVX2_M64: #define __XSAVEOPT__ 1
// CHECK_CORE_AVX2_M64: #define __XSAVE__ 1
// CHECK_CORE_AVX2_M64: #define __amd64 1
// CHECK_CORE_AVX2_M64: #define __amd64__ 1
// CHECK_CORE_AVX2_M64: #define __corei7 1
// CHECK_CORE_AVX2_M64: #define __corei7__ 1
// CHECK_CORE_AVX2_M64: #define __tune_corei7__ 1
// CHECK_CORE_AVX2_M64: #define __x86_64 1
// CHECK_CORE_AVX2_M64: #define __x86_64__ 1
//
// RUN: %clang -march=broadwell -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BROADWELL_M32
// CHECK_BROADWELL_M32: #define __ADX__ 1
// CHECK_BROADWELL_M32: #define __AES__ 1
// CHECK_BROADWELL_M32: #define __AVX2__ 1
// CHECK_BROADWELL_M32: #define __AVX__ 1
// CHECK_BROADWELL_M32: #define __BMI2__ 1
// CHECK_BROADWELL_M32: #define __BMI__ 1
// CHECK_BROADWELL_M32: #define __F16C__ 1
// CHECK_BROADWELL_M32: #define __FMA__ 1
// CHECK_BROADWELL_M32: #define __LZCNT__ 1
// CHECK_BROADWELL_M32: #define __MMX__ 1
// CHECK_BROADWELL_M32: #define __PCLMUL__ 1
// CHECK_BROADWELL_M32: #define __POPCNT__ 1
// CHECK_BROADWELL_M32: #define __RDRND__ 1
// CHECK_BROADWELL_M32: #define __RDSEED__ 1
// CHECK_BROADWELL_M32: #define __SSE2__ 1
// CHECK_BROADWELL_M32: #define __SSE3__ 1
// CHECK_BROADWELL_M32: #define __SSE4_1__ 1
// CHECK_BROADWELL_M32: #define __SSE4_2__ 1
// CHECK_BROADWELL_M32: #define __SSE__ 1
// CHECK_BROADWELL_M32: #define __SSSE3__ 1
// CHECK_BROADWELL_M32: #define __XSAVEOPT__ 1
// CHECK_BROADWELL_M32: #define __XSAVE__ 1
// CHECK_BROADWELL_M32: #define __corei7 1
// CHECK_BROADWELL_M32: #define __corei7__ 1
// CHECK_BROADWELL_M32: #define __i386 1
// CHECK_BROADWELL_M32: #define __i386__ 1
// CHECK_BROADWELL_M32: #define __tune_corei7__ 1
// CHECK_BROADWELL_M32: #define i386 1
// RUN: %clang -march=broadwell -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BROADWELL_M64
// CHECK_BROADWELL_M64: #define __ADX__ 1
// CHECK_BROADWELL_M64: #define __AES__ 1
// CHECK_BROADWELL_M64: #define __AVX2__ 1
// CHECK_BROADWELL_M64: #define __AVX__ 1
// CHECK_BROADWELL_M64: #define __BMI2__ 1
// CHECK_BROADWELL_M64: #define __BMI__ 1
// CHECK_BROADWELL_M64: #define __F16C__ 1
// CHECK_BROADWELL_M64: #define __FMA__ 1
// CHECK_BROADWELL_M64: #define __LZCNT__ 1
// CHECK_BROADWELL_M64: #define __MMX__ 1
// CHECK_BROADWELL_M64: #define __PCLMUL__ 1
// CHECK_BROADWELL_M64: #define __POPCNT__ 1
// CHECK_BROADWELL_M64: #define __RDRND__ 1
// CHECK_BROADWELL_M64: #define __RDSEED__ 1
// CHECK_BROADWELL_M64: #define __SSE2_MATH__ 1
// CHECK_BROADWELL_M64: #define __SSE2__ 1
// CHECK_BROADWELL_M64: #define __SSE3__ 1
// CHECK_BROADWELL_M64: #define __SSE4_1__ 1
// CHECK_BROADWELL_M64: #define __SSE4_2__ 1
// CHECK_BROADWELL_M64: #define __SSE_MATH__ 1
// CHECK_BROADWELL_M64: #define __SSE__ 1
// CHECK_BROADWELL_M64: #define __SSSE3__ 1
// CHECK_BROADWELL_M64: #define __XSAVEOPT__ 1
// CHECK_BROADWELL_M64: #define __XSAVE__ 1
// CHECK_BROADWELL_M64: #define __amd64 1
// CHECK_BROADWELL_M64: #define __amd64__ 1
// CHECK_BROADWELL_M64: #define __corei7 1
// CHECK_BROADWELL_M64: #define __corei7__ 1
// CHECK_BROADWELL_M64: #define __tune_corei7__ 1
// CHECK_BROADWELL_M64: #define __x86_64 1
// CHECK_BROADWELL_M64: #define __x86_64__ 1
//
// RUN: %clang -march=skylake -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SKL_M32
// CHECK_SKL_M32: #define __ADX__ 1
// CHECK_SKL_M32: #define __AES__ 1
// CHECK_SKL_M32: #define __AVX2__ 1
// CHECK_SKL_M32: #define __AVX__ 1
// CHECK_SKL_M32: #define __BMI2__ 1
// CHECK_SKL_M32: #define __BMI__ 1
// CHECK_SKL_M32: #define __CLFLUSHOPT__ 1
// CHECK_SKL_M32: #define __F16C__ 1
// CHECK_SKL_M32: #define __FMA__ 1
// CHECK_SKL_M32: #define __LZCNT__ 1
// CHECK_SKL_M32: #define __MMX__ 1
// CHECK_SKL_M32: #define __MPX__ 1
// CHECK_SKL_M32: #define __PCLMUL__ 1
// CHECK_SKL_M32: #define __POPCNT__ 1
// CHECK_SKL_M32: #define __RDRND__ 1
// CHECK_SKL_M32: #define __RDSEED__ 1
// CHECK_SKL_M32: #define __RTM__ 1
// CHECK_SKL_M32: #define __SGX__ 1
// CHECK_SKL_M32: #define __SSE2__ 1
// CHECK_SKL_M32: #define __SSE3__ 1
// CHECK_SKL_M32: #define __SSE4_1__ 1
// CHECK_SKL_M32: #define __SSE4_2__ 1
// CHECK_SKL_M32: #define __SSE__ 1
// CHECK_SKL_M32: #define __SSSE3__ 1
// CHECK_SKL_M32: #define __XSAVEC__ 1
// CHECK_SKL_M32: #define __XSAVEOPT__ 1
// CHECK_SKL_M32: #define __XSAVES__ 1
// CHECK_SKL_M32: #define __XSAVE__ 1
// CHECK_SKL_M32: #define i386 1

// RUN: %clang -march=skylake -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SKL_M64
// CHECK_SKL_M64: #define __ADX__ 1
// CHECK_SKL_M64: #define __AES__ 1
// CHECK_SKL_M64: #define __AVX2__ 1
// CHECK_SKL_M64: #define __AVX__ 1
// CHECK_SKL_M64: #define __BMI2__ 1
// CHECK_SKL_M64: #define __BMI__ 1
// CHECK_SKL_M64: #define __CLFLUSHOPT__ 1
// CHECK_SKL_M64: #define __F16C__ 1
// CHECK_SKL_M64: #define __FMA__ 1
// CHECK_SKL_M64: #define __LZCNT__ 1
// CHECK_SKL_M64: #define __MMX__ 1
// CHECK_SKL_M64: #define __MPX__ 1
// CHECK_SKL_M64: #define __PCLMUL__ 1
// CHECK_SKL_M64: #define __POPCNT__ 1
// CHECK_SKL_M64: #define __RDRND__ 1
// CHECK_SKL_M64: #define __RDSEED__ 1
// CHECK_SKL_M64: #define __RTM__ 1
// CHECK_SKL_M64: #define __SGX__ 1
// CHECK_SKL_M64: #define __SSE2_MATH__ 1
// CHECK_SKL_M64: #define __SSE2__ 1
// CHECK_SKL_M64: #define __SSE3__ 1
// CHECK_SKL_M64: #define __SSE4_1__ 1
// CHECK_SKL_M64: #define __SSE4_2__ 1
// CHECK_SKL_M64: #define __SSE_MATH__ 1
// CHECK_SKL_M64: #define __SSE__ 1
// CHECK_SKL_M64: #define __SSSE3__ 1
// CHECK_SKL_M64: #define __XSAVEC__ 1
// CHECK_SKL_M64: #define __XSAVEOPT__ 1
// CHECK_SKL_M64: #define __XSAVES__ 1
// CHECK_SKL_M64: #define __XSAVE__ 1
// CHECK_SKL_M64: #define __amd64 1
// CHECK_SKL_M64: #define __amd64__ 1
// CHECK_SKL_M64: #define __x86_64 1
// CHECK_SKL_M64: #define __x86_64__ 1

// RUN: %clang -march=knl -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_KNL_M32
// CHECK_KNL_M32: #define __AES__ 1
// CHECK_KNL_M32: #define __AVX2__ 1
// CHECK_KNL_M32: #define __AVX512CD__ 1
// CHECK_KNL_M32: #define __AVX512ER__ 1
// CHECK_KNL_M32: #define __AVX512F__ 1
// CHECK_KNL_M32: #define __AVX512PF__ 1
// CHECK_KNL_M32: #define __AVX__ 1
// CHECK_KNL_M32: #define __BMI2__ 1
// CHECK_KNL_M32: #define __BMI__ 1
// CHECK_KNL_M32: #define __F16C__ 1
// CHECK_KNL_M32: #define __FMA__ 1
// CHECK_KNL_M32: #define __LZCNT__ 1
// CHECK_KNL_M32: #define __MMX__ 1
// CHECK_KNL_M32: #define __PCLMUL__ 1
// CHECK_KNL_M32: #define __POPCNT__ 1
// CHECK_KNL_M32: #define __PREFETCHWT1__ 1
// CHECK_KNL_M32: #define __RDRND__ 1
// CHECK_KNL_M32: #define __RTM__ 1
// CHECK_KNL_M32: #define __SSE2__ 1
// CHECK_KNL_M32: #define __SSE3__ 1
// CHECK_KNL_M32: #define __SSE4_1__ 1
// CHECK_KNL_M32: #define __SSE4_2__ 1
// CHECK_KNL_M32: #define __SSE__ 1
// CHECK_KNL_M32: #define __SSSE3__ 1
// CHECK_KNL_M32: #define __XSAVEOPT__ 1
// CHECK_KNL_M32: #define __XSAVE__ 1
// CHECK_KNL_M32: #define __i386 1
// CHECK_KNL_M32: #define __i386__ 1
// CHECK_KNL_M32: #define __knl 1
// CHECK_KNL_M32: #define __knl__ 1
// CHECK_KNL_M32: #define __tune_knl__ 1
// CHECK_KNL_M32: #define i386 1

// RUN: %clang -march=knl -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_KNL_M64
// CHECK_KNL_M64: #define __AES__ 1
// CHECK_KNL_M64: #define __AVX2__ 1
// CHECK_KNL_M64: #define __AVX512CD__ 1
// CHECK_KNL_M64: #define __AVX512ER__ 1
// CHECK_KNL_M64: #define __AVX512F__ 1
// CHECK_KNL_M64: #define __AVX512PF__ 1
// CHECK_KNL_M64: #define __AVX__ 1
// CHECK_KNL_M64: #define __BMI2__ 1
// CHECK_KNL_M64: #define __BMI__ 1
// CHECK_KNL_M64: #define __F16C__ 1
// CHECK_KNL_M64: #define __FMA__ 1
// CHECK_KNL_M64: #define __LZCNT__ 1
// CHECK_KNL_M64: #define __MMX__ 1
// CHECK_KNL_M64: #define __PCLMUL__ 1
// CHECK_KNL_M64: #define __POPCNT__ 1
// CHECK_KNL_M64: #define __PREFETCHWT1__ 1
// CHECK_KNL_M64: #define __RDRND__ 1
// CHECK_KNL_M64: #define __RTM__ 1
// CHECK_KNL_M64: #define __SSE2_MATH__ 1
// CHECK_KNL_M64: #define __SSE2__ 1
// CHECK_KNL_M64: #define __SSE3__ 1
// CHECK_KNL_M64: #define __SSE4_1__ 1
// CHECK_KNL_M64: #define __SSE4_2__ 1
// CHECK_KNL_M64: #define __SSE_MATH__ 1
// CHECK_KNL_M64: #define __SSE__ 1
// CHECK_KNL_M64: #define __SSSE3__ 1
// CHECK_KNL_M64: #define __XSAVEOPT__ 1
// CHECK_KNL_M64: #define __XSAVE__ 1
// CHECK_KNL_M64: #define __amd64 1
// CHECK_KNL_M64: #define __amd64__ 1
// CHECK_KNL_M64: #define __knl 1
// CHECK_KNL_M64: #define __knl__ 1
// CHECK_KNL_M64: #define __tune_knl__ 1
// CHECK_KNL_M64: #define __x86_64 1
// CHECK_KNL_M64: #define __x86_64__ 1

// RUN: %clang -march=knm -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_KNM_M32
// CHECK_KNM_M32: #define __AES__ 1
// CHECK_KNM_M32: #define __AVX2__ 1
// CHECK_KNM_M32: #define __AVX512CD__ 1
// CHECK_KNM_M32: #define __AVX512ER__ 1
// CHECK_KNM_M32: #define __AVX512F__ 1
// CHECK_KNM_M32: #define __AVX512PF__ 1
// CHECK_KNM_M32: #define __AVX__ 1
// CHECK_KNM_M32: #define __BMI2__ 1
// CHECK_KNM_M32: #define __BMI__ 1
// CHECK_KNM_M32: #define __F16C__ 1
// CHECK_KNM_M32: #define __FMA__ 1
// CHECK_KNM_M32: #define __LZCNT__ 1
// CHECK_KNM_M32: #define __MMX__ 1
// CHECK_KNM_M32: #define __PCLMUL__ 1
// CHECK_KNM_M32: #define __POPCNT__ 1
// CHECK_KNM_M32: #define __PREFETCHWT1__ 1
// CHECK_KNM_M32: #define __RDRND__ 1
// CHECK_KNM_M32: #define __RTM__ 1
// CHECK_KNM_M32: #define __SSE2__ 1
// CHECK_KNM_M32: #define __SSE3__ 1
// CHECK_KNM_M32: #define __SSE4_1__ 1
// CHECK_KNM_M32: #define __SSE4_2__ 1
// CHECK_KNM_M32: #define __SSE__ 1
// CHECK_KNM_M32: #define __SSSE3__ 1
// CHECK_KNM_M32: #define __XSAVEOPT__ 1
// CHECK_KNM_M32: #define __XSAVE__ 1
// CHECK_KNM_M32: #define __i386 1
// CHECK_KNM_M32: #define __i386__ 1
// CHECK_KNM_M32: #define i386 1

// RUN: %clang -march=knm -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_KNM_M64
// CHECK_KNM_M64: #define __AES__ 1
// CHECK_KNM_M64: #define __AVX2__ 1
// CHECK_KNM_M64: #define __AVX512CD__ 1
// CHECK_KNM_M64: #define __AVX512ER__ 1
// CHECK_KNM_M64: #define __AVX512F__ 1
// CHECK_KNM_M64: #define __AVX512PF__ 1
// CHECK_KNM_M64: #define __AVX__ 1
// CHECK_KNM_M64: #define __BMI2__ 1
// CHECK_KNM_M64: #define __BMI__ 1
// CHECK_KNM_M64: #define __F16C__ 1
// CHECK_KNM_M64: #define __FMA__ 1
// CHECK_KNM_M64: #define __LZCNT__ 1
// CHECK_KNM_M64: #define __MMX__ 1
// CHECK_KNM_M64: #define __PCLMUL__ 1
// CHECK_KNM_M64: #define __POPCNT__ 1
// CHECK_KNM_M64: #define __PREFETCHWT1__ 1
// CHECK_KNM_M64: #define __RDRND__ 1
// CHECK_KNM_M64: #define __RTM__ 1
// CHECK_KNM_M64: #define __SSE2_MATH__ 1
// CHECK_KNM_M64: #define __SSE2__ 1
// CHECK_KNM_M64: #define __SSE3__ 1
// CHECK_KNM_M64: #define __SSE4_1__ 1
// CHECK_KNM_M64: #define __SSE4_2__ 1
// CHECK_KNM_M64: #define __SSE_MATH__ 1
// CHECK_KNM_M64: #define __SSE__ 1
// CHECK_KNM_M64: #define __SSSE3__ 1
// CHECK_KNM_M64: #define __XSAVEOPT__ 1
// CHECK_KNM_M64: #define __XSAVE__ 1
// CHECK_KNM_M64: #define __amd64 1
// CHECK_KNM_M64: #define __amd64__ 1
// CHECK_KNM_M64: #define __x86_64 1
// CHECK_KNM_M64: #define __x86_64__ 1
//
// RUN: %clang -march=skylake-avx512 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SKX_M32
// CHECK_SKX_M32: #define __AES__ 1
// CHECK_SKX_M32: #define __AVX2__ 1
// CHECK_SKX_M32: #define __AVX512BW__ 1
// CHECK_SKX_M32: #define __AVX512CD__ 1
// CHECK_SKX_M32: #define __AVX512DQ__ 1
// CHECK_SKX_M32: #define __AVX512F__ 1
// CHECK_SKX_M32: #define __AVX512VL__ 1
// CHECK_SKX_M32: #define __AVX__ 1
// CHECK_SKX_M32: #define __BMI2__ 1
// CHECK_SKX_M32: #define __BMI__ 1
// CHECK_SKX_M32: #define __CLFLUSHOPT__ 1
// CHECK_SKX_M32: #define __CLWB__ 1
// CHECK_SKX_M32: #define __F16C__ 1
// CHECK_SKX_M32: #define __FMA__ 1
// CHECK_SKX_M32: #define __LZCNT__ 1
// CHECK_SKX_M32: #define __MMX__ 1
// CHECK_SKX_M32: #define __MPX__ 1
// CHECK_SKX_M32: #define __PCLMUL__ 1
// CHECK_SKX_M32: #define __POPCNT__ 1
// CHECK_SKX_M32: #define __RDRND__ 1
// CHECK_SKX_M32: #define __RTM__ 1
// CHECK_SKX_M32: #define __SGX__ 1
// CHECK_SKX_M32: #define __SSE2__ 1
// CHECK_SKX_M32: #define __SSE3__ 1
// CHECK_SKX_M32: #define __SSE4_1__ 1
// CHECK_SKX_M32: #define __SSE4_2__ 1
// CHECK_SKX_M32: #define __SSE__ 1
// CHECK_SKX_M32: #define __SSSE3__ 1
// CHECK_SKX_M32: #define __XSAVEC__ 1
// CHECK_SKX_M32: #define __XSAVEOPT__ 1
// CHECK_SKX_M32: #define __XSAVES__ 1
// CHECK_SKX_M32: #define __XSAVE__ 1
// CHECK_SKX_M32: #define __i386 1
// CHECK_SKX_M32: #define __i386__ 1
// CHECK_SKX_M32: #define __skx 1
// CHECK_SKX_M32: #define __skx__ 1
// CHECK_SKX_M32: #define __tune_skx__ 1
// CHECK_SKX_M32: #define i386 1

// RUN: %clang -march=skylake-avx512 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SKX_M64
// CHECK_SKX_M64: #define __AES__ 1
// CHECK_SKX_M64: #define __AVX2__ 1
// CHECK_SKX_M64: #define __AVX512BW__ 1
// CHECK_SKX_M64: #define __AVX512CD__ 1
// CHECK_SKX_M64: #define __AVX512DQ__ 1
// CHECK_SKX_M64: #define __AVX512F__ 1
// CHECK_SKX_M64: #define __AVX512VL__ 1
// CHECK_SKX_M64: #define __AVX__ 1
// CHECK_SKX_M64: #define __BMI2__ 1
// CHECK_SKX_M64: #define __BMI__ 1
// CHECK_SKX_M64: #define __CLFLUSHOPT__ 1
// CHECK_SKX_M64: #define __CLWB__ 1
// CHECK_SKX_M64: #define __F16C__ 1
// CHECK_SKX_M64: #define __FMA__ 1
// CHECK_SKX_M64: #define __LZCNT__ 1
// CHECK_SKX_M64: #define __MMX__ 1
// CHECK_SKX_M64: #define __MPX__ 1
// CHECK_SKX_M64: #define __PCLMUL__ 1
// CHECK_SKX_M64: #define __POPCNT__ 1
// CHECK_SKX_M64: #define __RDRND__ 1
// CHECK_SKX_M64: #define __RTM__ 1
// CHECK_SKX_M64: #define __SGX__ 1
// CHECK_SKX_M64: #define __SSE2_MATH__ 1
// CHECK_SKX_M64: #define __SSE2__ 1
// CHECK_SKX_M64: #define __SSE3__ 1
// CHECK_SKX_M64: #define __SSE4_1__ 1
// CHECK_SKX_M64: #define __SSE4_2__ 1
// CHECK_SKX_M64: #define __SSE_MATH__ 1
// CHECK_SKX_M64: #define __SSE__ 1
// CHECK_SKX_M64: #define __SSSE3__ 1
// CHECK_SKX_M64: #define __XSAVEC__ 1
// CHECK_SKX_M64: #define __XSAVEOPT__ 1
// CHECK_SKX_M64: #define __XSAVES__ 1
// CHECK_SKX_M64: #define __XSAVE__ 1
// CHECK_SKX_M64: #define __amd64 1
// CHECK_SKX_M64: #define __amd64__ 1
// CHECK_SKX_M64: #define __skx 1
// CHECK_SKX_M64: #define __skx__ 1
// CHECK_SKX_M64: #define __tune_skx__ 1
// CHECK_SKX_M64: #define __x86_64 1
// CHECK_SKX_M64: #define __x86_64__ 1
//
// RUN: %clang -march=cannonlake -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_CNL_M32
// CHECK_CNL_M32: #define __AES__ 1
// CHECK_CNL_M32: #define __AVX2__ 1
// CHECK_CNL_M32: #define __AVX512BW__ 1
// CHECK_CNL_M32: #define __AVX512CD__ 1
// CHECK_CNL_M32: #define __AVX512DQ__ 1
// CHECK_CNL_M32: #define __AVX512F__ 1
// CHECK_CNL_M32: #define __AVX512IFMA__ 1
// CHECK_CNL_M32: #define __AVX512VBMI__ 1
// CHECK_CNL_M32: #define __AVX512VL__ 1
// CHECK_CNL_M32: #define __AVX__ 1
// CHECK_CNL_M32: #define __BMI2__ 1
// CHECK_CNL_M32: #define __BMI__ 1
// CHECK_CNL_M32: #define __CLFLUSHOPT__ 1
// CHECK_CNL_M32: #define __F16C__ 1
// CHECK_CNL_M32: #define __FMA__ 1
// CHECK_CNL_M32: #define __LZCNT__ 1
// CHECK_CNL_M32: #define __MMX__ 1
// CHECK_CNL_M32: #define __MPX__ 1
// CHECK_CNL_M32: #define __PCLMUL__ 1
// CHECK_CNL_M32: #define __POPCNT__ 1
// CHECK_CNL_M32: #define __RDRND__ 1
// CHECK_CNL_M32: #define __RTM__ 1
// CHECK_CNL_M32: #define __SGX__ 1
// CHECK_CNL_M32: #define __SHA__ 1
// CHECK_CNL_M32: #define __SSE2__ 1
// CHECK_CNL_M32: #define __SSE3__ 1
// CHECK_CNL_M32: #define __SSE4_1__ 1
// CHECK_CNL_M32: #define __SSE4_2__ 1
// CHECK_CNL_M32: #define __SSE__ 1
// CHECK_CNL_M32: #define __SSSE3__ 1
// CHECK_CNL_M32: #define __XSAVEC__ 1
// CHECK_CNL_M32: #define __XSAVEOPT__ 1
// CHECK_CNL_M32: #define __XSAVES__ 1
// CHECK_CNL_M32: #define __XSAVE__ 1
// CHECK_CNL_M32: #define __i386 1
// CHECK_CNL_M32: #define __i386__ 1
// CHECK_CNL_M32: #define i386 1
//
// RUN: %clang -march=cannonlake -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_CNL_M64
// CHECK_CNL_M64: #define __AES__ 1
// CHECK_CNL_M64: #define __AVX2__ 1
// CHECK_CNL_M64: #define __AVX512BW__ 1
// CHECK_CNL_M64: #define __AVX512CD__ 1
// CHECK_CNL_M64: #define __AVX512DQ__ 1
// CHECK_CNL_M64: #define __AVX512F__ 1
// CHECK_CNL_M64: #define __AVX512IFMA__ 1
// CHECK_CNL_M64: #define __AVX512VBMI__ 1
// CHECK_CNL_M64: #define __AVX512VL__ 1
// CHECK_CNL_M64: #define __AVX__ 1
// CHECK_CNL_M64: #define __BMI2__ 1
// CHECK_CNL_M64: #define __BMI__ 1
// CHECK_CNL_M64: #define __CLFLUSHOPT__ 1
// CHECK_CNL_M64: #define __F16C__ 1
// CHECK_CNL_M64: #define __FMA__ 1
// CHECK_CNL_M64: #define __LZCNT__ 1
// CHECK_CNL_M64: #define __MMX__ 1
// CHECK_CNL_M64: #define __MPX__ 1
// CHECK_CNL_M64: #define __PCLMUL__ 1
// CHECK_CNL_M64: #define __POPCNT__ 1
// CHECK_CNL_M64: #define __RDRND__ 1
// CHECK_CNL_M64: #define __RTM__ 1
// CHECK_CNL_M64: #define __SGX__ 1
// CHECK_CNL_M64: #define __SHA__ 1
// CHECK_CNL_M64: #define __SSE2__ 1
// CHECK_CNL_M64: #define __SSE3__ 1
// CHECK_CNL_M64: #define __SSE4_1__ 1
// CHECK_CNL_M64: #define __SSE4_2__ 1
// CHECK_CNL_M64: #define __SSE__ 1
// CHECK_CNL_M64: #define __SSSE3__ 1
// CHECK_CNL_M64: #define __XSAVEC__ 1
// CHECK_CNL_M64: #define __XSAVEOPT__ 1
// CHECK_CNL_M64: #define __XSAVES__ 1
// CHECK_CNL_M64: #define __XSAVE__ 1
// CHECK_CNL_M64: #define __amd64 1
// CHECK_CNL_M64: #define __amd64__ 1
// CHECK_CNL_M64: #define __x86_64 1
// CHECK_CNL_M64: #define __x86_64__ 1

// RUN: %clang -march=atom -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATOM_M32
// CHECK_ATOM_M32: #define __MMX__ 1
// CHECK_ATOM_M32: #define __SSE2__ 1
// CHECK_ATOM_M32: #define __SSE3__ 1
// CHECK_ATOM_M32: #define __SSE__ 1
// CHECK_ATOM_M32: #define __SSSE3__ 1
// CHECK_ATOM_M32: #define __atom 1
// CHECK_ATOM_M32: #define __atom__ 1
// CHECK_ATOM_M32: #define __i386 1
// CHECK_ATOM_M32: #define __i386__ 1
// CHECK_ATOM_M32: #define __tune_atom__ 1
// CHECK_ATOM_M32: #define i386 1
// RUN: %clang -march=atom -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATOM_M64
// CHECK_ATOM_M64: #define __MMX__ 1
// CHECK_ATOM_M64: #define __SSE2_MATH__ 1
// CHECK_ATOM_M64: #define __SSE2__ 1
// CHECK_ATOM_M64: #define __SSE3__ 1
// CHECK_ATOM_M64: #define __SSE_MATH__ 1
// CHECK_ATOM_M64: #define __SSE__ 1
// CHECK_ATOM_M64: #define __SSSE3__ 1
// CHECK_ATOM_M64: #define __amd64 1
// CHECK_ATOM_M64: #define __amd64__ 1
// CHECK_ATOM_M64: #define __atom 1
// CHECK_ATOM_M64: #define __atom__ 1
// CHECK_ATOM_M64: #define __tune_atom__ 1
// CHECK_ATOM_M64: #define __x86_64 1
// CHECK_ATOM_M64: #define __x86_64__ 1
//
// RUN: %clang -march=goldmont -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck %s -check-prefix=CHECK_GLM_M32
// CHECK_GLM_M32: #define __AES__ 1
// CHECK_GLM_M32: #define __CLFLUSHOPT__ 1
// CHECK_GLM_M32: #define __FSGSBASE__ 1
// CHECK_GLM_M32: #define __FXSR__ 1
// CHECK_GLM_M32: #define __MMX__ 1
// CHECK_GLM_M32: #define __MPX__ 1
// CHECK_GLM_M32: #define __PCLMUL__ 1
// CHECK_GLM_M32: #define __POPCNT__ 1
// CHECK_GLM_M32: #define __RDRND__ 1
// CHECK_GLM_M32: #define __RDSEED__ 1
// CHECK_GLM_M32: #define __SHA__ 1
// CHECK_GLM_M32: #define __SSE2__ 1
// CHECK_GLM_M32: #define __SSE3__ 1
// CHECK_GLM_M32: #define __SSE4_1__ 1
// CHECK_GLM_M32: #define __SSE4_2__ 1
// CHECK_GLM_M32: #define __SSE_MATH__ 1
// CHECK_GLM_M32: #define __SSE__ 1
// CHECK_GLM_M32: #define __SSSE3__ 1
// CHECK_GLM_M32: #define __XSAVEC__ 1
// CHECK_GLM_M32: #define __XSAVEOPT__ 1
// CHECK_GLM_M32: #define __XSAVES__ 1
// CHECK_GLM_M32: #define __XSAVE__ 1
// CHECK_GLM_M32: #define __goldmont 1
// CHECK_GLM_M32: #define __goldmont__ 1
// CHECK_GLM_M32: #define __i386 1
// CHECK_GLM_M32: #define __i386__ 1
// CHECK_GLM_M32: #define __tune_goldmont__ 1
// CHECK_GLM_M32: #define i386 1
//
// RUN: %clang -march=goldmont -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck %s -check-prefix=CHECK_GLM_M64
// CHECK_GLM_M64: #define __AES__ 1
// CHECK_GLM_M64: #define __CLFLUSHOPT__ 1
// CHECK_GLM_M64: #define __FSGSBASE__ 1
// CHECK_GLM_M64: #define __FXSR__ 1
// CHECK_GLM_M64: #define __MMX__ 1
// CHECK_GLM_M64: #define __MPX__ 1
// CHECK_GLM_M64: #define __PCLMUL__ 1
// CHECK_GLM_M64: #define __POPCNT__ 1
// CHECK_GLM_M64: #define __RDRND__ 1
// CHECK_GLM_M64: #define __RDSEED__ 1
// CHECK_GLM_M64: #define __SSE2__ 1
// CHECK_GLM_M64: #define __SSE3__ 1
// CHECK_GLM_M64: #define __SSE4_1__ 1
// CHECK_GLM_M64: #define __SSE4_2__ 1
// CHECK_GLM_M64: #define __SSE__ 1
// CHECK_GLM_M64: #define __SSSE3__ 1
// CHECK_GLM_M64: #define __XSAVEC__ 1
// CHECK_GLM_M64: #define __XSAVEOPT__ 1
// CHECK_GLM_M64: #define __XSAVES__ 1
// CHECK_GLM_M64: #define __XSAVE__ 1
// CHECK_GLM_M64: #define __goldmont 1
// CHECK_GLM_M64: #define __goldmont__ 1
// CHECK_GLM_M64: #define __tune_goldmont__ 1
// CHECK_GLM_M64: #define __x86_64 1
// CHECK_GLM_M64: #define __x86_64__ 1
//
// RUN: %clang -march=slm -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SLM_M32
// CHECK_SLM_M32: #define __MMX__ 1
// CHECK_SLM_M32: #define __SSE2__ 1
// CHECK_SLM_M32: #define __SSE3__ 1
// CHECK_SLM_M32: #define __SSE4_1__ 1
// CHECK_SLM_M32: #define __SSE4_2__ 1
// CHECK_SLM_M32: #define __SSE__ 1
// CHECK_SLM_M32: #define __SSSE3__ 1
// CHECK_SLM_M32: #define __i386 1
// CHECK_SLM_M32: #define __i386__ 1
// CHECK_SLM_M32: #define __slm 1
// CHECK_SLM_M32: #define __slm__ 1
// CHECK_SLM_M32: #define __tune_slm__ 1
// CHECK_SLM_M32: #define i386 1
// RUN: %clang -march=slm -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SLM_M64
// CHECK_SLM_M64: #define __MMX__ 1
// CHECK_SLM_M64: #define __SSE2_MATH__ 1
// CHECK_SLM_M64: #define __SSE2__ 1
// CHECK_SLM_M64: #define __SSE3__ 1
// CHECK_SLM_M64: #define __SSE4_1__ 1
// CHECK_SLM_M64: #define __SSE4_2__ 1
// CHECK_SLM_M64: #define __SSE_MATH__ 1
// CHECK_SLM_M64: #define __SSE__ 1
// CHECK_SLM_M64: #define __SSSE3__ 1
// CHECK_SLM_M64: #define __amd64 1
// CHECK_SLM_M64: #define __amd64__ 1
// CHECK_SLM_M64: #define __slm 1
// CHECK_SLM_M64: #define __slm__ 1
// CHECK_SLM_M64: #define __tune_slm__ 1
// CHECK_SLM_M64: #define __x86_64 1
// CHECK_SLM_M64: #define __x86_64__ 1
//
// RUN: %clang -march=lakemont -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck %s -check-prefix=CHECK_LMT_M32
// CHECK_LMT_M32: #define __i386 1
// CHECK_LMT_M32: #define __i386__ 1
// CHECK_LMT_M32: #define __tune_lakemont__ 1
// CHECK_LMT_M32: #define i386 1
// RUN: not %clang -march=lakemont -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck %s -check-prefix=CHECK_LMT_M64
// CHECK_LMT_M64: error:
//
// RUN: %clang -march=geode -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_GEODE_M32
// CHECK_GEODE_M32: #define __3dNOW_A__ 1
// CHECK_GEODE_M32: #define __3dNOW__ 1
// CHECK_GEODE_M32: #define __MMX__ 1
// CHECK_GEODE_M32: #define __geode 1
// CHECK_GEODE_M32: #define __geode__ 1
// CHECK_GEODE_M32: #define __i386 1
// CHECK_GEODE_M32: #define __i386__ 1
// CHECK_GEODE_M32: #define __tune_geode__ 1
// CHECK_GEODE_M32: #define i386 1
// RUN: not %clang -march=geode -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_GEODE_M64
// CHECK_GEODE_M64: error: {{.*}}
//
// RUN: %clang -march=k6 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_K6_M32
// CHECK_K6_M32: #define __MMX__ 1
// CHECK_K6_M32: #define __i386 1
// CHECK_K6_M32: #define __i386__ 1
// CHECK_K6_M32: #define __k6 1
// CHECK_K6_M32: #define __k6__ 1
// CHECK_K6_M32: #define __tune_k6__ 1
// CHECK_K6_M32: #define i386 1
// RUN: not %clang -march=k6 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_K6_M64
// CHECK_K6_M64: error: {{.*}}
//
// RUN: %clang -march=k6-2 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_K6_2_M32
// CHECK_K6_2_M32: #define __3dNOW__ 1
// CHECK_K6_2_M32: #define __MMX__ 1
// CHECK_K6_2_M32: #define __i386 1
// CHECK_K6_2_M32: #define __i386__ 1
// CHECK_K6_2_M32: #define __k6 1
// CHECK_K6_2_M32: #define __k6_2__ 1
// CHECK_K6_2_M32: #define __k6__ 1
// CHECK_K6_2_M32: #define __tune_k6_2__ 1
// CHECK_K6_2_M32: #define __tune_k6__ 1
// CHECK_K6_2_M32: #define i386 1
// RUN: not %clang -march=k6-2 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_K6_2_M64
// CHECK_K6_2_M64: error: {{.*}}
//
// RUN: %clang -march=k6-3 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_K6_3_M32
// CHECK_K6_3_M32: #define __3dNOW__ 1
// CHECK_K6_3_M32: #define __MMX__ 1
// CHECK_K6_3_M32: #define __i386 1
// CHECK_K6_3_M32: #define __i386__ 1
// CHECK_K6_3_M32: #define __k6 1
// CHECK_K6_3_M32: #define __k6_3__ 1
// CHECK_K6_3_M32: #define __k6__ 1
// CHECK_K6_3_M32: #define __tune_k6_3__ 1
// CHECK_K6_3_M32: #define __tune_k6__ 1
// CHECK_K6_3_M32: #define i386 1
// RUN: not %clang -march=k6-3 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_K6_3_M64
// CHECK_K6_3_M64: error: {{.*}}
//
// RUN: %clang -march=athlon -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON_M32
// CHECK_ATHLON_M32: #define __3dNOW_A__ 1
// CHECK_ATHLON_M32: #define __3dNOW__ 1
// CHECK_ATHLON_M32: #define __MMX__ 1
// CHECK_ATHLON_M32: #define __athlon 1
// CHECK_ATHLON_M32: #define __athlon__ 1
// CHECK_ATHLON_M32: #define __i386 1
// CHECK_ATHLON_M32: #define __i386__ 1
// CHECK_ATHLON_M32: #define __tune_athlon__ 1
// CHECK_ATHLON_M32: #define i386 1
// RUN: not %clang -march=athlon -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON_M64
// CHECK_ATHLON_M64: error: {{.*}}
//
// RUN: %clang -march=athlon-tbird -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON_TBIRD_M32
// CHECK_ATHLON_TBIRD_M32: #define __3dNOW_A__ 1
// CHECK_ATHLON_TBIRD_M32: #define __3dNOW__ 1
// CHECK_ATHLON_TBIRD_M32: #define __MMX__ 1
// CHECK_ATHLON_TBIRD_M32: #define __athlon 1
// CHECK_ATHLON_TBIRD_M32: #define __athlon__ 1
// CHECK_ATHLON_TBIRD_M32: #define __i386 1
// CHECK_ATHLON_TBIRD_M32: #define __i386__ 1
// CHECK_ATHLON_TBIRD_M32: #define __tune_athlon__ 1
// CHECK_ATHLON_TBIRD_M32: #define i386 1
// RUN: not %clang -march=athlon-tbird -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON_TBIRD_M64
// CHECK_ATHLON_TBIRD_M64: error: {{.*}}
//
// RUN: %clang -march=athlon-4 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON_4_M32
// CHECK_ATHLON_4_M32: #define __3dNOW_A__ 1
// CHECK_ATHLON_4_M32: #define __3dNOW__ 1
// CHECK_ATHLON_4_M32: #define __MMX__ 1
// CHECK_ATHLON_4_M32: #define __SSE__ 1
// CHECK_ATHLON_4_M32: #define __athlon 1
// CHECK_ATHLON_4_M32: #define __athlon__ 1
// CHECK_ATHLON_4_M32: #define __athlon_sse__ 1
// CHECK_ATHLON_4_M32: #define __i386 1
// CHECK_ATHLON_4_M32: #define __i386__ 1
// CHECK_ATHLON_4_M32: #define __tune_athlon__ 1
// CHECK_ATHLON_4_M32: #define __tune_athlon_sse__ 1
// CHECK_ATHLON_4_M32: #define i386 1
// RUN: not %clang -march=athlon-4 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON_4_M64
// CHECK_ATHLON_4_M64: error: {{.*}}
//
// RUN: %clang -march=athlon-xp -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON_XP_M32
// CHECK_ATHLON_XP_M32: #define __3dNOW_A__ 1
// CHECK_ATHLON_XP_M32: #define __3dNOW__ 1
// CHECK_ATHLON_XP_M32: #define __MMX__ 1
// CHECK_ATHLON_XP_M32: #define __SSE__ 1
// CHECK_ATHLON_XP_M32: #define __athlon 1
// CHECK_ATHLON_XP_M32: #define __athlon__ 1
// CHECK_ATHLON_XP_M32: #define __athlon_sse__ 1
// CHECK_ATHLON_XP_M32: #define __i386 1
// CHECK_ATHLON_XP_M32: #define __i386__ 1
// CHECK_ATHLON_XP_M32: #define __tune_athlon__ 1
// CHECK_ATHLON_XP_M32: #define __tune_athlon_sse__ 1
// CHECK_ATHLON_XP_M32: #define i386 1
// RUN: not %clang -march=athlon-xp -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON_XP_M64
// CHECK_ATHLON_XP_M64: error: {{.*}}
//
// RUN: %clang -march=athlon-mp -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON_MP_M32
// CHECK_ATHLON_MP_M32: #define __3dNOW_A__ 1
// CHECK_ATHLON_MP_M32: #define __3dNOW__ 1
// CHECK_ATHLON_MP_M32: #define __MMX__ 1
// CHECK_ATHLON_MP_M32: #define __SSE__ 1
// CHECK_ATHLON_MP_M32: #define __athlon 1
// CHECK_ATHLON_MP_M32: #define __athlon__ 1
// CHECK_ATHLON_MP_M32: #define __athlon_sse__ 1
// CHECK_ATHLON_MP_M32: #define __i386 1
// CHECK_ATHLON_MP_M32: #define __i386__ 1
// CHECK_ATHLON_MP_M32: #define __tune_athlon__ 1
// CHECK_ATHLON_MP_M32: #define __tune_athlon_sse__ 1
// CHECK_ATHLON_MP_M32: #define i386 1
// RUN: not %clang -march=athlon-mp -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON_MP_M64
// CHECK_ATHLON_MP_M64: error: {{.*}}
//
// RUN: %clang -march=x86-64 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_X86_64_M32
// CHECK_X86_64_M32: #define __MMX__ 1
// CHECK_X86_64_M32: #define __SSE2__ 1
// CHECK_X86_64_M32: #define __SSE__ 1
// CHECK_X86_64_M32: #define __i386 1
// CHECK_X86_64_M32: #define __i386__ 1
// CHECK_X86_64_M32: #define __k8 1
// CHECK_X86_64_M32: #define __k8__ 1
// CHECK_X86_64_M32: #define i386 1
// RUN: %clang -march=x86-64 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_X86_64_M64
// CHECK_X86_64_M64: #define __MMX__ 1
// CHECK_X86_64_M64: #define __SSE2_MATH__ 1
// CHECK_X86_64_M64: #define __SSE2__ 1
// CHECK_X86_64_M64: #define __SSE_MATH__ 1
// CHECK_X86_64_M64: #define __SSE__ 1
// CHECK_X86_64_M64: #define __amd64 1
// CHECK_X86_64_M64: #define __amd64__ 1
// CHECK_X86_64_M64: #define __k8 1
// CHECK_X86_64_M64: #define __k8__ 1
// CHECK_X86_64_M64: #define __x86_64 1
// CHECK_X86_64_M64: #define __x86_64__ 1
//
// RUN: %clang -march=k8 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_K8_M32
// CHECK_K8_M32: #define __3dNOW_A__ 1
// CHECK_K8_M32: #define __3dNOW__ 1
// CHECK_K8_M32: #define __MMX__ 1
// CHECK_K8_M32: #define __SSE2__ 1
// CHECK_K8_M32: #define __SSE__ 1
// CHECK_K8_M32: #define __i386 1
// CHECK_K8_M32: #define __i386__ 1
// CHECK_K8_M32: #define __k8 1
// CHECK_K8_M32: #define __k8__ 1
// CHECK_K8_M32: #define __tune_k8__ 1
// CHECK_K8_M32: #define i386 1
// RUN: %clang -march=k8 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_K8_M64
// CHECK_K8_M64: #define __3dNOW_A__ 1
// CHECK_K8_M64: #define __3dNOW__ 1
// CHECK_K8_M64: #define __MMX__ 1
// CHECK_K8_M64: #define __SSE2_MATH__ 1
// CHECK_K8_M64: #define __SSE2__ 1
// CHECK_K8_M64: #define __SSE_MATH__ 1
// CHECK_K8_M64: #define __SSE__ 1
// CHECK_K8_M64: #define __amd64 1
// CHECK_K8_M64: #define __amd64__ 1
// CHECK_K8_M64: #define __k8 1
// CHECK_K8_M64: #define __k8__ 1
// CHECK_K8_M64: #define __tune_k8__ 1
// CHECK_K8_M64: #define __x86_64 1
// CHECK_K8_M64: #define __x86_64__ 1
//
// RUN: %clang -march=k8-sse3 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_K8_SSE3_M32
// CHECK_K8_SSE3_M32: #define __3dNOW_A__ 1
// CHECK_K8_SSE3_M32: #define __3dNOW__ 1
// CHECK_K8_SSE3_M32: #define __MMX__ 1
// CHECK_K8_SSE3_M32: #define __SSE2__ 1
// CHECK_K8_SSE3_M32: #define __SSE3__ 1
// CHECK_K8_SSE3_M32: #define __SSE__ 1
// CHECK_K8_SSE3_M32: #define __i386 1
// CHECK_K8_SSE3_M32: #define __i386__ 1
// CHECK_K8_SSE3_M32: #define __k8 1
// CHECK_K8_SSE3_M32: #define __k8__ 1
// CHECK_K8_SSE3_M32: #define __tune_k8__ 1
// CHECK_K8_SSE3_M32: #define i386 1
// RUN: %clang -march=k8-sse3 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_K8_SSE3_M64
// CHECK_K8_SSE3_M64: #define __3dNOW_A__ 1
// CHECK_K8_SSE3_M64: #define __3dNOW__ 1
// CHECK_K8_SSE3_M64: #define __MMX__ 1
// CHECK_K8_SSE3_M64: #define __SSE2_MATH__ 1
// CHECK_K8_SSE3_M64: #define __SSE2__ 1
// CHECK_K8_SSE3_M64: #define __SSE3__ 1
// CHECK_K8_SSE3_M64: #define __SSE_MATH__ 1
// CHECK_K8_SSE3_M64: #define __SSE__ 1
// CHECK_K8_SSE3_M64: #define __amd64 1
// CHECK_K8_SSE3_M64: #define __amd64__ 1
// CHECK_K8_SSE3_M64: #define __k8 1
// CHECK_K8_SSE3_M64: #define __k8__ 1
// CHECK_K8_SSE3_M64: #define __tune_k8__ 1
// CHECK_K8_SSE3_M64: #define __x86_64 1
// CHECK_K8_SSE3_M64: #define __x86_64__ 1
//
// RUN: %clang -march=opteron -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_OPTERON_M32
// CHECK_OPTERON_M32: #define __3dNOW_A__ 1
// CHECK_OPTERON_M32: #define __3dNOW__ 1
// CHECK_OPTERON_M32: #define __MMX__ 1
// CHECK_OPTERON_M32: #define __SSE2__ 1
// CHECK_OPTERON_M32: #define __SSE__ 1
// CHECK_OPTERON_M32: #define __i386 1
// CHECK_OPTERON_M32: #define __i386__ 1
// CHECK_OPTERON_M32: #define __k8 1
// CHECK_OPTERON_M32: #define __k8__ 1
// CHECK_OPTERON_M32: #define __tune_k8__ 1
// CHECK_OPTERON_M32: #define i386 1
// RUN: %clang -march=opteron -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_OPTERON_M64
// CHECK_OPTERON_M64: #define __3dNOW_A__ 1
// CHECK_OPTERON_M64: #define __3dNOW__ 1
// CHECK_OPTERON_M64: #define __MMX__ 1
// CHECK_OPTERON_M64: #define __SSE2_MATH__ 1
// CHECK_OPTERON_M64: #define __SSE2__ 1
// CHECK_OPTERON_M64: #define __SSE_MATH__ 1
// CHECK_OPTERON_M64: #define __SSE__ 1
// CHECK_OPTERON_M64: #define __amd64 1
// CHECK_OPTERON_M64: #define __amd64__ 1
// CHECK_OPTERON_M64: #define __k8 1
// CHECK_OPTERON_M64: #define __k8__ 1
// CHECK_OPTERON_M64: #define __tune_k8__ 1
// CHECK_OPTERON_M64: #define __x86_64 1
// CHECK_OPTERON_M64: #define __x86_64__ 1
//
// RUN: %clang -march=opteron-sse3 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_OPTERON_SSE3_M32
// CHECK_OPTERON_SSE3_M32: #define __3dNOW_A__ 1
// CHECK_OPTERON_SSE3_M32: #define __3dNOW__ 1
// CHECK_OPTERON_SSE3_M32: #define __MMX__ 1
// CHECK_OPTERON_SSE3_M32: #define __SSE2__ 1
// CHECK_OPTERON_SSE3_M32: #define __SSE3__ 1
// CHECK_OPTERON_SSE3_M32: #define __SSE__ 1
// CHECK_OPTERON_SSE3_M32: #define __i386 1
// CHECK_OPTERON_SSE3_M32: #define __i386__ 1
// CHECK_OPTERON_SSE3_M32: #define __k8 1
// CHECK_OPTERON_SSE3_M32: #define __k8__ 1
// CHECK_OPTERON_SSE3_M32: #define __tune_k8__ 1
// CHECK_OPTERON_SSE3_M32: #define i386 1
// RUN: %clang -march=opteron-sse3 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_OPTERON_SSE3_M64
// CHECK_OPTERON_SSE3_M64: #define __3dNOW_A__ 1
// CHECK_OPTERON_SSE3_M64: #define __3dNOW__ 1
// CHECK_OPTERON_SSE3_M64: #define __MMX__ 1
// CHECK_OPTERON_SSE3_M64: #define __SSE2_MATH__ 1
// CHECK_OPTERON_SSE3_M64: #define __SSE2__ 1
// CHECK_OPTERON_SSE3_M64: #define __SSE3__ 1
// CHECK_OPTERON_SSE3_M64: #define __SSE_MATH__ 1
// CHECK_OPTERON_SSE3_M64: #define __SSE__ 1
// CHECK_OPTERON_SSE3_M64: #define __amd64 1
// CHECK_OPTERON_SSE3_M64: #define __amd64__ 1
// CHECK_OPTERON_SSE3_M64: #define __k8 1
// CHECK_OPTERON_SSE3_M64: #define __k8__ 1
// CHECK_OPTERON_SSE3_M64: #define __tune_k8__ 1
// CHECK_OPTERON_SSE3_M64: #define __x86_64 1
// CHECK_OPTERON_SSE3_M64: #define __x86_64__ 1
//
// RUN: %clang -march=athlon64 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON64_M32
// CHECK_ATHLON64_M32: #define __3dNOW_A__ 1
// CHECK_ATHLON64_M32: #define __3dNOW__ 1
// CHECK_ATHLON64_M32: #define __MMX__ 1
// CHECK_ATHLON64_M32: #define __SSE2__ 1
// CHECK_ATHLON64_M32: #define __SSE__ 1
// CHECK_ATHLON64_M32: #define __i386 1
// CHECK_ATHLON64_M32: #define __i386__ 1
// CHECK_ATHLON64_M32: #define __k8 1
// CHECK_ATHLON64_M32: #define __k8__ 1
// CHECK_ATHLON64_M32: #define __tune_k8__ 1
// CHECK_ATHLON64_M32: #define i386 1
// RUN: %clang -march=athlon64 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON64_M64
// CHECK_ATHLON64_M64: #define __3dNOW_A__ 1
// CHECK_ATHLON64_M64: #define __3dNOW__ 1
// CHECK_ATHLON64_M64: #define __MMX__ 1
// CHECK_ATHLON64_M64: #define __SSE2_MATH__ 1
// CHECK_ATHLON64_M64: #define __SSE2__ 1
// CHECK_ATHLON64_M64: #define __SSE_MATH__ 1
// CHECK_ATHLON64_M64: #define __SSE__ 1
// CHECK_ATHLON64_M64: #define __amd64 1
// CHECK_ATHLON64_M64: #define __amd64__ 1
// CHECK_ATHLON64_M64: #define __k8 1
// CHECK_ATHLON64_M64: #define __k8__ 1
// CHECK_ATHLON64_M64: #define __tune_k8__ 1
// CHECK_ATHLON64_M64: #define __x86_64 1
// CHECK_ATHLON64_M64: #define __x86_64__ 1
//
// RUN: %clang -march=athlon64-sse3 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON64_SSE3_M32
// CHECK_ATHLON64_SSE3_M32: #define __3dNOW_A__ 1
// CHECK_ATHLON64_SSE3_M32: #define __3dNOW__ 1
// CHECK_ATHLON64_SSE3_M32: #define __MMX__ 1
// CHECK_ATHLON64_SSE3_M32: #define __SSE2__ 1
// CHECK_ATHLON64_SSE3_M32: #define __SSE3__ 1
// CHECK_ATHLON64_SSE3_M32: #define __SSE__ 1
// CHECK_ATHLON64_SSE3_M32: #define __i386 1
// CHECK_ATHLON64_SSE3_M32: #define __i386__ 1
// CHECK_ATHLON64_SSE3_M32: #define __k8 1
// CHECK_ATHLON64_SSE3_M32: #define __k8__ 1
// CHECK_ATHLON64_SSE3_M32: #define __tune_k8__ 1
// CHECK_ATHLON64_SSE3_M32: #define i386 1
// RUN: %clang -march=athlon64-sse3 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON64_SSE3_M64
// CHECK_ATHLON64_SSE3_M64: #define __3dNOW_A__ 1
// CHECK_ATHLON64_SSE3_M64: #define __3dNOW__ 1
// CHECK_ATHLON64_SSE3_M64: #define __MMX__ 1
// CHECK_ATHLON64_SSE3_M64: #define __SSE2_MATH__ 1
// CHECK_ATHLON64_SSE3_M64: #define __SSE2__ 1
// CHECK_ATHLON64_SSE3_M64: #define __SSE3__ 1
// CHECK_ATHLON64_SSE3_M64: #define __SSE_MATH__ 1
// CHECK_ATHLON64_SSE3_M64: #define __SSE__ 1
// CHECK_ATHLON64_SSE3_M64: #define __amd64 1
// CHECK_ATHLON64_SSE3_M64: #define __amd64__ 1
// CHECK_ATHLON64_SSE3_M64: #define __k8 1
// CHECK_ATHLON64_SSE3_M64: #define __k8__ 1
// CHECK_ATHLON64_SSE3_M64: #define __tune_k8__ 1
// CHECK_ATHLON64_SSE3_M64: #define __x86_64 1
// CHECK_ATHLON64_SSE3_M64: #define __x86_64__ 1
//
// RUN: %clang -march=athlon-fx -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON_FX_M32
// CHECK_ATHLON_FX_M32: #define __3dNOW_A__ 1
// CHECK_ATHLON_FX_M32: #define __3dNOW__ 1
// CHECK_ATHLON_FX_M32: #define __MMX__ 1
// CHECK_ATHLON_FX_M32: #define __SSE2__ 1
// CHECK_ATHLON_FX_M32: #define __SSE__ 1
// CHECK_ATHLON_FX_M32: #define __i386 1
// CHECK_ATHLON_FX_M32: #define __i386__ 1
// CHECK_ATHLON_FX_M32: #define __k8 1
// CHECK_ATHLON_FX_M32: #define __k8__ 1
// CHECK_ATHLON_FX_M32: #define __tune_k8__ 1
// CHECK_ATHLON_FX_M32: #define i386 1
// RUN: %clang -march=athlon-fx -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ATHLON_FX_M64
// CHECK_ATHLON_FX_M64: #define __3dNOW_A__ 1
// CHECK_ATHLON_FX_M64: #define __3dNOW__ 1
// CHECK_ATHLON_FX_M64: #define __MMX__ 1
// CHECK_ATHLON_FX_M64: #define __SSE2_MATH__ 1
// CHECK_ATHLON_FX_M64: #define __SSE2__ 1
// CHECK_ATHLON_FX_M64: #define __SSE_MATH__ 1
// CHECK_ATHLON_FX_M64: #define __SSE__ 1
// CHECK_ATHLON_FX_M64: #define __amd64 1
// CHECK_ATHLON_FX_M64: #define __amd64__ 1
// CHECK_ATHLON_FX_M64: #define __k8 1
// CHECK_ATHLON_FX_M64: #define __k8__ 1
// CHECK_ATHLON_FX_M64: #define __tune_k8__ 1
// CHECK_ATHLON_FX_M64: #define __x86_64 1
// CHECK_ATHLON_FX_M64: #define __x86_64__ 1
// RUN: %clang -march=amdfam10 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_AMDFAM10_M32
// CHECK_AMDFAM10_M32: #define __3dNOW_A__ 1
// CHECK_AMDFAM10_M32: #define __3dNOW__ 1
// CHECK_AMDFAM10_M32: #define __LZCNT__ 1
// CHECK_AMDFAM10_M32: #define __MMX__ 1
// CHECK_AMDFAM10_M32: #define __POPCNT__ 1
// CHECK_AMDFAM10_M32: #define __SSE2_MATH__ 1
// CHECK_AMDFAM10_M32: #define __SSE2__ 1
// CHECK_AMDFAM10_M32: #define __SSE3__ 1
// CHECK_AMDFAM10_M32: #define __SSE4A__ 1
// CHECK_AMDFAM10_M32: #define __SSE_MATH__ 1
// CHECK_AMDFAM10_M32: #define __SSE__ 1
// CHECK_AMDFAM10_M32: #define __amdfam10 1
// CHECK_AMDFAM10_M32: #define __amdfam10__ 1
// CHECK_AMDFAM10_M32: #define __i386 1
// CHECK_AMDFAM10_M32: #define __i386__ 1
// CHECK_AMDFAM10_M32: #define __tune_amdfam10__ 1
// RUN: %clang -march=amdfam10 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_AMDFAM10_M64
// CHECK_AMDFAM10_M64: #define __3dNOW_A__ 1
// CHECK_AMDFAM10_M64: #define __3dNOW__ 1
// CHECK_AMDFAM10_M64: #define __LZCNT__ 1
// CHECK_AMDFAM10_M64: #define __MMX__ 1
// CHECK_AMDFAM10_M64: #define __POPCNT__ 1
// CHECK_AMDFAM10_M64: #define __SSE2_MATH__ 1
// CHECK_AMDFAM10_M64: #define __SSE2__ 1
// CHECK_AMDFAM10_M64: #define __SSE3__ 1
// CHECK_AMDFAM10_M64: #define __SSE4A__ 1
// CHECK_AMDFAM10_M64: #define __SSE_MATH__ 1
// CHECK_AMDFAM10_M64: #define __SSE__ 1
// CHECK_AMDFAM10_M64: #define __amd64 1
// CHECK_AMDFAM10_M64: #define __amd64__ 1
// CHECK_AMDFAM10_M64: #define __amdfam10 1
// CHECK_AMDFAM10_M64: #define __amdfam10__ 1
// CHECK_AMDFAM10_M64: #define __tune_amdfam10__ 1
// CHECK_AMDFAM10_M64: #define __x86_64 1
// CHECK_AMDFAM10_M64: #define __x86_64__ 1
// RUN: %clang -march=btver1 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BTVER1_M32
// CHECK_BTVER1_M32-NOT: #define __3dNOW_A__ 1
// CHECK_BTVER1_M32-NOT: #define __3dNOW__ 1
// CHECK_BTVER1_M32: #define __LZCNT__ 1
// CHECK_BTVER1_M32: #define __MMX__ 1
// CHECK_BTVER1_M32: #define __POPCNT__ 1
// CHECK_BTVER1_M32: #define __PRFCHW__ 1
// CHECK_BTVER1_M32: #define __SSE2_MATH__ 1
// CHECK_BTVER1_M32: #define __SSE2__ 1
// CHECK_BTVER1_M32: #define __SSE3__ 1
// CHECK_BTVER1_M32: #define __SSE4A__ 1
// CHECK_BTVER1_M32: #define __SSE_MATH__ 1
// CHECK_BTVER1_M32: #define __SSE__ 1
// CHECK_BTVER1_M32: #define __SSSE3__ 1
// CHECK_BTVER1_M32: #define __btver1 1
// CHECK_BTVER1_M32: #define __btver1__ 1
// CHECK_BTVER1_M32: #define __i386 1
// CHECK_BTVER1_M32: #define __i386__ 1
// CHECK_BTVER1_M32: #define __tune_btver1__ 1
// RUN: %clang -march=btver1 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BTVER1_M64
// CHECK_BTVER1_M64-NOT: #define __3dNOW_A__ 1
// CHECK_BTVER1_M64-NOT: #define __3dNOW__ 1
// CHECK_BTVER1_M64: #define __LZCNT__ 1
// CHECK_BTVER1_M64: #define __MMX__ 1
// CHECK_BTVER1_M64: #define __POPCNT__ 1
// CHECK_BTVER1_M64: #define __PRFCHW__ 1
// CHECK_BTVER1_M64: #define __SSE2_MATH__ 1
// CHECK_BTVER1_M64: #define __SSE2__ 1
// CHECK_BTVER1_M64: #define __SSE3__ 1
// CHECK_BTVER1_M64: #define __SSE4A__ 1
// CHECK_BTVER1_M64: #define __SSE_MATH__ 1
// CHECK_BTVER1_M64: #define __SSE__ 1
// CHECK_BTVER1_M64: #define __SSSE3__ 1
// CHECK_BTVER1_M64: #define __amd64 1
// CHECK_BTVER1_M64: #define __amd64__ 1
// CHECK_BTVER1_M64: #define __btver1 1
// CHECK_BTVER1_M64: #define __btver1__ 1
// CHECK_BTVER1_M64: #define __tune_btver1__ 1
// CHECK_BTVER1_M64: #define __x86_64 1
// CHECK_BTVER1_M64: #define __x86_64__ 1
// RUN: %clang -march=btver2 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BTVER2_M32
// CHECK_BTVER2_M32-NOT: #define __3dNOW_A__ 1
// CHECK_BTVER2_M32-NOT: #define __3dNOW__ 1
// CHECK_BTVER2_M32: #define __AES__ 1
// CHECK_BTVER2_M32: #define __AVX__ 1
// CHECK_BTVER2_M32: #define __BMI__ 1
// CHECK_BTVER2_M32: #define __F16C__ 1
// CHECK_BTVER2_M32: #define __LZCNT__ 1
// CHECK_BTVER2_M32: #define __MMX__ 1
// CHECK_BTVER2_M32: #define __PCLMUL__ 1
// CHECK_BTVER2_M32: #define __POPCNT__ 1
// CHECK_BTVER2_M32: #define __PRFCHW__ 1
// CHECK_BTVER2_M32: #define __SSE2_MATH__ 1
// CHECK_BTVER2_M32: #define __SSE2__ 1
// CHECK_BTVER2_M32: #define __SSE3__ 1
// CHECK_BTVER2_M32: #define __SSE4A__ 1
// CHECK_BTVER2_M32: #define __SSE_MATH__ 1
// CHECK_BTVER2_M32: #define __SSE__ 1
// CHECK_BTVER2_M32: #define __SSSE3__ 1
// CHECK_BTVER2_M32: #define __XSAVEOPT__ 1
// CHECK_BTVER2_M32: #define __XSAVE__ 1
// CHECK_BTVER2_M32: #define __btver2 1
// CHECK_BTVER2_M32: #define __btver2__ 1
// CHECK_BTVER2_M32: #define __i386 1
// CHECK_BTVER2_M32: #define __i386__ 1
// CHECK_BTVER2_M32: #define __tune_btver2__ 1
// RUN: %clang -march=btver2 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BTVER2_M64
// CHECK_BTVER2_M64-NOT: #define __3dNOW_A__ 1
// CHECK_BTVER2_M64-NOT: #define __3dNOW__ 1
// CHECK_BTVER2_M64: #define __AES__ 1
// CHECK_BTVER2_M64: #define __AVX__ 1
// CHECK_BTVER2_M64: #define __BMI__ 1
// CHECK_BTVER2_M64: #define __F16C__ 1
// CHECK_BTVER2_M64: #define __LZCNT__ 1
// CHECK_BTVER2_M64: #define __MMX__ 1
// CHECK_BTVER2_M64: #define __PCLMUL__ 1
// CHECK_BTVER2_M64: #define __POPCNT__ 1
// CHECK_BTVER2_M64: #define __PRFCHW__ 1
// CHECK_BTVER2_M64: #define __SSE2_MATH__ 1
// CHECK_BTVER2_M64: #define __SSE2__ 1
// CHECK_BTVER2_M64: #define __SSE3__ 1
// CHECK_BTVER2_M64: #define __SSE4A__ 1
// CHECK_BTVER2_M64: #define __SSE_MATH__ 1
// CHECK_BTVER2_M64: #define __SSE__ 1
// CHECK_BTVER2_M64: #define __SSSE3__ 1
// CHECK_BTVER2_M64: #define __XSAVEOPT__ 1
// CHECK_BTVER2_M64: #define __XSAVE__ 1
// CHECK_BTVER2_M64: #define __amd64 1
// CHECK_BTVER2_M64: #define __amd64__ 1
// CHECK_BTVER2_M64: #define __btver2 1
// CHECK_BTVER2_M64: #define __btver2__ 1
// CHECK_BTVER2_M64: #define __tune_btver2__ 1
// CHECK_BTVER2_M64: #define __x86_64 1
// CHECK_BTVER2_M64: #define __x86_64__ 1
// RUN: %clang -march=bdver1 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BDVER1_M32
// CHECK_BDVER1_M32-NOT: #define __3dNOW_A__ 1
// CHECK_BDVER1_M32-NOT: #define __3dNOW__ 1
// CHECK_BDVER1_M32: #define __AES__ 1
// CHECK_BDVER1_M32: #define __AVX__ 1
// CHECK_BDVER1_M32: #define __FMA4__ 1
// CHECK_BDVER1_M32: #define __LWP__ 1
// CHECK_BDVER1_M32: #define __LZCNT__ 1
// CHECK_BDVER1_M32: #define __MMX__ 1
// CHECK_BDVER1_M32: #define __PCLMUL__ 1
// CHECK_BDVER1_M32: #define __POPCNT__ 1
// CHECK_BDVER1_M32: #define __PRFCHW__ 1
// CHECK_BDVER1_M32: #define __SSE2_MATH__ 1
// CHECK_BDVER1_M32: #define __SSE2__ 1
// CHECK_BDVER1_M32: #define __SSE3__ 1
// CHECK_BDVER1_M32: #define __SSE4A__ 1
// CHECK_BDVER1_M32: #define __SSE4_1__ 1
// CHECK_BDVER1_M32: #define __SSE4_2__ 1
// CHECK_BDVER1_M32: #define __SSE_MATH__ 1
// CHECK_BDVER1_M32: #define __SSE__ 1
// CHECK_BDVER1_M32: #define __SSSE3__ 1
// CHECK_BDVER1_M32: #define __XOP__ 1
// CHECK_BDVER1_M32: #define __XSAVE__ 1
// CHECK_BDVER1_M32: #define __bdver1 1
// CHECK_BDVER1_M32: #define __bdver1__ 1
// CHECK_BDVER1_M32: #define __i386 1
// CHECK_BDVER1_M32: #define __i386__ 1
// CHECK_BDVER1_M32: #define __tune_bdver1__ 1
// RUN: %clang -march=bdver1 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BDVER1_M64
// CHECK_BDVER1_M64-NOT: #define __3dNOW_A__ 1
// CHECK_BDVER1_M64-NOT: #define __3dNOW__ 1
// CHECK_BDVER1_M64: #define __AES__ 1
// CHECK_BDVER1_M64: #define __AVX__ 1
// CHECK_BDVER1_M64: #define __FMA4__ 1
// CHECK_BDVER1_M64: #define __LWP__ 1
// CHECK_BDVER1_M64: #define __LZCNT__ 1
// CHECK_BDVER1_M64: #define __MMX__ 1
// CHECK_BDVER1_M64: #define __PCLMUL__ 1
// CHECK_BDVER1_M64: #define __POPCNT__ 1
// CHECK_BDVER1_M64: #define __PRFCHW__ 1
// CHECK_BDVER1_M64: #define __SSE2_MATH__ 1
// CHECK_BDVER1_M64: #define __SSE2__ 1
// CHECK_BDVER1_M64: #define __SSE3__ 1
// CHECK_BDVER1_M64: #define __SSE4A__ 1
// CHECK_BDVER1_M64: #define __SSE4_1__ 1
// CHECK_BDVER1_M64: #define __SSE4_2__ 1
// CHECK_BDVER1_M64: #define __SSE_MATH__ 1
// CHECK_BDVER1_M64: #define __SSE__ 1
// CHECK_BDVER1_M64: #define __SSSE3__ 1
// CHECK_BDVER1_M64: #define __XOP__ 1
// CHECK_BDVER1_M64: #define __XSAVE__ 1
// CHECK_BDVER1_M64: #define __amd64 1
// CHECK_BDVER1_M64: #define __amd64__ 1
// CHECK_BDVER1_M64: #define __bdver1 1
// CHECK_BDVER1_M64: #define __bdver1__ 1
// CHECK_BDVER1_M64: #define __tune_bdver1__ 1
// CHECK_BDVER1_M64: #define __x86_64 1
// CHECK_BDVER1_M64: #define __x86_64__ 1
// RUN: %clang -march=bdver2 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BDVER2_M32
// CHECK_BDVER2_M32-NOT: #define __3dNOW_A__ 1
// CHECK_BDVER2_M32-NOT: #define __3dNOW__ 1
// CHECK_BDVER2_M32: #define __AES__ 1
// CHECK_BDVER2_M32: #define __AVX__ 1
// CHECK_BDVER2_M32: #define __BMI__ 1
// CHECK_BDVER2_M32: #define __F16C__ 1
// CHECK_BDVER2_M32: #define __FMA4__ 1
// CHECK_BDVER2_M32: #define __FMA__ 1
// CHECK_BDVER2_M32: #define __LWP__ 1
// CHECK_BDVER2_M32: #define __LZCNT__ 1
// CHECK_BDVER2_M32: #define __MMX__ 1
// CHECK_BDVER2_M32: #define __PCLMUL__ 1
// CHECK_BDVER2_M32: #define __POPCNT__ 1
// CHECK_BDVER2_M32: #define __PRFCHW__ 1
// CHECK_BDVER2_M32: #define __SSE2_MATH__ 1
// CHECK_BDVER2_M32: #define __SSE2__ 1
// CHECK_BDVER2_M32: #define __SSE3__ 1
// CHECK_BDVER2_M32: #define __SSE4A__ 1
// CHECK_BDVER2_M32: #define __SSE4_1__ 1
// CHECK_BDVER2_M32: #define __SSE4_2__ 1
// CHECK_BDVER2_M32: #define __SSE_MATH__ 1
// CHECK_BDVER2_M32: #define __SSE__ 1
// CHECK_BDVER2_M32: #define __SSSE3__ 1
// CHECK_BDVER2_M32: #define __TBM__ 1
// CHECK_BDVER2_M32: #define __XOP__ 1
// CHECK_BDVER2_M32: #define __XSAVE__ 1
// CHECK_BDVER2_M32: #define __bdver2 1
// CHECK_BDVER2_M32: #define __bdver2__ 1
// CHECK_BDVER2_M32: #define __i386 1
// CHECK_BDVER2_M32: #define __i386__ 1
// CHECK_BDVER2_M32: #define __tune_bdver2__ 1
// RUN: %clang -march=bdver2 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BDVER2_M64
// CHECK_BDVER2_M64-NOT: #define __3dNOW_A__ 1
// CHECK_BDVER2_M64-NOT: #define __3dNOW__ 1
// CHECK_BDVER2_M64: #define __AES__ 1
// CHECK_BDVER2_M64: #define __AVX__ 1
// CHECK_BDVER2_M64: #define __BMI__ 1
// CHECK_BDVER2_M64: #define __F16C__ 1
// CHECK_BDVER2_M64: #define __FMA4__ 1
// CHECK_BDVER2_M64: #define __FMA__ 1
// CHECK_BDVER2_M64: #define __LWP__ 1
// CHECK_BDVER2_M64: #define __LZCNT__ 1
// CHECK_BDVER2_M64: #define __MMX__ 1
// CHECK_BDVER2_M64: #define __PCLMUL__ 1
// CHECK_BDVER2_M64: #define __POPCNT__ 1
// CHECK_BDVER2_M64: #define __PRFCHW__ 1
// CHECK_BDVER2_M64: #define __SSE2_MATH__ 1
// CHECK_BDVER2_M64: #define __SSE2__ 1
// CHECK_BDVER2_M64: #define __SSE3__ 1
// CHECK_BDVER2_M64: #define __SSE4A__ 1
// CHECK_BDVER2_M64: #define __SSE4_1__ 1
// CHECK_BDVER2_M64: #define __SSE4_2__ 1
// CHECK_BDVER2_M64: #define __SSE_MATH__ 1
// CHECK_BDVER2_M64: #define __SSE__ 1
// CHECK_BDVER2_M64: #define __SSSE3__ 1
// CHECK_BDVER2_M64: #define __TBM__ 1
// CHECK_BDVER2_M64: #define __XOP__ 1
// CHECK_BDVER2_M64: #define __XSAVE__ 1
// CHECK_BDVER2_M64: #define __amd64 1
// CHECK_BDVER2_M64: #define __amd64__ 1
// CHECK_BDVER2_M64: #define __bdver2 1
// CHECK_BDVER2_M64: #define __bdver2__ 1
// CHECK_BDVER2_M64: #define __tune_bdver2__ 1
// CHECK_BDVER2_M64: #define __x86_64 1
// CHECK_BDVER2_M64: #define __x86_64__ 1
// RUN: %clang -march=bdver3 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BDVER3_M32
// CHECK_BDVER3_M32-NOT: #define __3dNOW_A__ 1
// CHECK_BDVER3_M32-NOT: #define __3dNOW__ 1
// CHECK_BDVER3_M32: #define __AES__ 1
// CHECK_BDVER3_M32: #define __AVX__ 1
// CHECK_BDVER3_M32: #define __BMI__ 1
// CHECK_BDVER3_M32: #define __F16C__ 1
// CHECK_BDVER3_M32: #define __FMA4__ 1
// CHECK_BDVER3_M32: #define __FMA__ 1
// CHECK_BDVER3_M32: #define __FSGSBASE__ 1
// CHECK_BDVER3_M32: #define __LWP__ 1
// CHECK_BDVER3_M32: #define __LZCNT__ 1
// CHECK_BDVER3_M32: #define __MMX__ 1
// CHECK_BDVER3_M32: #define __PCLMUL__ 1
// CHECK_BDVER3_M32: #define __POPCNT__ 1
// CHECK_BDVER3_M32: #define __PRFCHW__ 1
// CHECK_BDVER3_M32: #define __SSE2_MATH__ 1
// CHECK_BDVER3_M32: #define __SSE2__ 1
// CHECK_BDVER3_M32: #define __SSE3__ 1
// CHECK_BDVER3_M32: #define __SSE4A__ 1
// CHECK_BDVER3_M32: #define __SSE4_1__ 1
// CHECK_BDVER3_M32: #define __SSE4_2__ 1
// CHECK_BDVER3_M32: #define __SSE_MATH__ 1
// CHECK_BDVER3_M32: #define __SSE__ 1
// CHECK_BDVER3_M32: #define __SSSE3__ 1
// CHECK_BDVER3_M32: #define __TBM__ 1
// CHECK_BDVER3_M32: #define __XOP__ 1
// CHECK_BDVER3_M32: #define __XSAVEOPT__ 1
// CHECK_BDVER3_M32: #define __XSAVE__ 1
// CHECK_BDVER3_M32: #define __bdver3 1
// CHECK_BDVER3_M32: #define __bdver3__ 1
// CHECK_BDVER3_M32: #define __i386 1
// CHECK_BDVER3_M32: #define __i386__ 1
// CHECK_BDVER3_M32: #define __tune_bdver3__ 1
// RUN: %clang -march=bdver3 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BDVER3_M64
// CHECK_BDVER3_M64-NOT: #define __3dNOW_A__ 1
// CHECK_BDVER3_M64-NOT: #define __3dNOW__ 1
// CHECK_BDVER3_M64: #define __AES__ 1
// CHECK_BDVER3_M64: #define __AVX__ 1
// CHECK_BDVER3_M64: #define __BMI__ 1
// CHECK_BDVER3_M64: #define __F16C__ 1
// CHECK_BDVER3_M64: #define __FMA4__ 1
// CHECK_BDVER3_M64: #define __FMA__ 1
// CHECK_BDVER3_M64: #define __FSGSBASE__ 1
// CHECK_BDVER3_M64: #define __LWP__ 1
// CHECK_BDVER3_M64: #define __LZCNT__ 1
// CHECK_BDVER3_M64: #define __MMX__ 1
// CHECK_BDVER3_M64: #define __PCLMUL__ 1
// CHECK_BDVER3_M64: #define __POPCNT__ 1
// CHECK_BDVER3_M64: #define __PRFCHW__ 1
// CHECK_BDVER3_M64: #define __SSE2_MATH__ 1
// CHECK_BDVER3_M64: #define __SSE2__ 1
// CHECK_BDVER3_M64: #define __SSE3__ 1
// CHECK_BDVER3_M64: #define __SSE4A__ 1
// CHECK_BDVER3_M64: #define __SSE4_1__ 1
// CHECK_BDVER3_M64: #define __SSE4_2__ 1
// CHECK_BDVER3_M64: #define __SSE_MATH__ 1
// CHECK_BDVER3_M64: #define __SSE__ 1
// CHECK_BDVER3_M64: #define __SSSE3__ 1
// CHECK_BDVER3_M64: #define __TBM__ 1
// CHECK_BDVER3_M64: #define __XOP__ 1
// CHECK_BDVER3_M64: #define __XSAVEOPT__ 1
// CHECK_BDVER3_M64: #define __XSAVE__ 1
// CHECK_BDVER3_M64: #define __amd64 1
// CHECK_BDVER3_M64: #define __amd64__ 1
// CHECK_BDVER3_M64: #define __bdver3 1
// CHECK_BDVER3_M64: #define __bdver3__ 1
// CHECK_BDVER3_M64: #define __tune_bdver3__ 1
// CHECK_BDVER3_M64: #define __x86_64 1
// CHECK_BDVER3_M64: #define __x86_64__ 1
// RUN: %clang -march=bdver4 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BDVER4_M32
// CHECK_BDVER4_M32-NOT: #define __3dNOW_A__ 1
// CHECK_BDVER4_M32-NOT: #define __3dNOW__ 1
// CHECK_BDVER4_M32: #define __AES__ 1
// CHECK_BDVER4_M32: #define __AVX2__ 1
// CHECK_BDVER4_M32: #define __AVX__ 1
// CHECK_BDVER4_M32: #define __BMI2__ 1
// CHECK_BDVER4_M32: #define __BMI__ 1
// CHECK_BDVER4_M32: #define __F16C__ 1
// CHECK_BDVER4_M32: #define __FMA4__ 1
// CHECK_BDVER4_M32: #define __FMA__ 1
// CHECK_BDVER4_M32: #define __FSGSBASE__ 1
// CHECK_BDVER4_M32: #define __LWP__ 1
// CHECK_BDVER4_M32: #define __LZCNT__ 1
// CHECK_BDVER4_M32: #define __MMX__ 1
// CHECK_BDVER4_M32: #define __PCLMUL__ 1
// CHECK_BDVER4_M32: #define __POPCNT__ 1
// CHECK_BDVER4_M32: #define __PRFCHW__ 1
// CHECK_BDVER4_M32: #define __SSE2_MATH__ 1
// CHECK_BDVER4_M32: #define __SSE2__ 1
// CHECK_BDVER4_M32: #define __SSE3__ 1
// CHECK_BDVER4_M32: #define __SSE4A__ 1
// CHECK_BDVER4_M32: #define __SSE4_1__ 1
// CHECK_BDVER4_M32: #define __SSE4_2__ 1
// CHECK_BDVER4_M32: #define __SSE_MATH__ 1
// CHECK_BDVER4_M32: #define __SSE__ 1
// CHECK_BDVER4_M32: #define __SSSE3__ 1
// CHECK_BDVER4_M32: #define __TBM__ 1
// CHECK_BDVER4_M32: #define __XOP__ 1
// CHECK_BDVER4_M32: #define __XSAVE__ 1
// CHECK_BDVER4_M32: #define __bdver4 1
// CHECK_BDVER4_M32: #define __bdver4__ 1
// CHECK_BDVER4_M32: #define __i386 1
// CHECK_BDVER4_M32: #define __i386__ 1
// CHECK_BDVER4_M32: #define __tune_bdver4__ 1
// RUN: %clang -march=bdver4 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_BDVER4_M64
// CHECK_BDVER4_M64-NOT: #define __3dNOW_A__ 1
// CHECK_BDVER4_M64-NOT: #define __3dNOW__ 1
// CHECK_BDVER4_M64: #define __AES__ 1
// CHECK_BDVER4_M64: #define __AVX2__ 1
// CHECK_BDVER4_M64: #define __AVX__ 1
// CHECK_BDVER4_M64: #define __BMI2__ 1
// CHECK_BDVER4_M64: #define __BMI__ 1
// CHECK_BDVER4_M64: #define __F16C__ 1
// CHECK_BDVER4_M64: #define __FMA4__ 1
// CHECK_BDVER4_M64: #define __FMA__ 1
// CHECK_BDVER4_M64: #define __FSGSBASE__ 1
// CHECK_BDVER4_M64: #define __LWP__ 1
// CHECK_BDVER4_M64: #define __LZCNT__ 1
// CHECK_BDVER4_M64: #define __MMX__ 1
// CHECK_BDVER4_M64: #define __PCLMUL__ 1
// CHECK_BDVER4_M64: #define __POPCNT__ 1
// CHECK_BDVER4_M64: #define __PRFCHW__ 1
// CHECK_BDVER4_M64: #define __SSE2_MATH__ 1
// CHECK_BDVER4_M64: #define __SSE2__ 1
// CHECK_BDVER4_M64: #define __SSE3__ 1
// CHECK_BDVER4_M64: #define __SSE4A__ 1
// CHECK_BDVER4_M64: #define __SSE4_1__ 1
// CHECK_BDVER4_M64: #define __SSE4_2__ 1
// CHECK_BDVER4_M64: #define __SSE_MATH__ 1
// CHECK_BDVER4_M64: #define __SSE__ 1
// CHECK_BDVER4_M64: #define __SSSE3__ 1
// CHECK_BDVER4_M64: #define __TBM__ 1
// CHECK_BDVER4_M64: #define __XOP__ 1
// CHECK_BDVER4_M64: #define __XSAVE__ 1
// CHECK_BDVER4_M64: #define __amd64 1
// CHECK_BDVER4_M64: #define __amd64__ 1
// CHECK_BDVER4_M64: #define __bdver4 1
// CHECK_BDVER4_M64: #define __bdver4__ 1
// CHECK_BDVER4_M64: #define __tune_bdver4__ 1
// CHECK_BDVER4_M64: #define __x86_64 1
// CHECK_BDVER4_M64: #define __x86_64__ 1
// RUN: %clang -march=znver1 -m32 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ZNVER1_M32
// CHECK_ZNVER1_M32-NOT: #define __3dNOW_A__ 1
// CHECK_ZNVER1_M32-NOT: #define __3dNOW__ 1
// CHECK_ZNVER1_M32: #define __ADX__ 1
// CHECK_ZNVER1_M32: #define __AES__ 1
// CHECK_ZNVER1_M32: #define __AVX2__ 1
// CHECK_ZNVER1_M32: #define __AVX__ 1
// CHECK_ZNVER1_M32: #define __BMI2__ 1
// CHECK_ZNVER1_M32: #define __BMI__ 1
// CHECK_ZNVER1_M32: #define __CLFLUSHOPT__ 1
// CHECK_ZNVER1_M32: #define __CLZERO__ 1
// CHECK_ZNVER1_M32: #define __F16C__ 1
// CHECK_ZNVER1_M32: #define __FMA__ 1
// CHECK_ZNVER1_M32: #define __FSGSBASE__ 1
// CHECK_ZNVER1_M32: #define __LZCNT__ 1
// CHECK_ZNVER1_M32: #define __MMX__ 1
// CHECK_ZNVER1_M32: #define __PCLMUL__ 1
// CHECK_ZNVER1_M32: #define __POPCNT__ 1
// CHECK_ZNVER1_M32: #define __PRFCHW__ 1
// CHECK_ZNVER1_M32: #define __RDRND__ 1
// CHECK_ZNVER1_M32: #define __RDSEED__ 1
// CHECK_ZNVER1_M32: #define __SHA__ 1
// CHECK_ZNVER1_M32: #define __SSE2_MATH__ 1
// CHECK_ZNVER1_M32: #define __SSE2__ 1
// CHECK_ZNVER1_M32: #define __SSE3__ 1
// CHECK_ZNVER1_M32: #define __SSE4A__ 1
// CHECK_ZNVER1_M32: #define __SSE4_1__ 1
// CHECK_ZNVER1_M32: #define __SSE4_2__ 1
// CHECK_ZNVER1_M32: #define __SSE_MATH__ 1
// CHECK_ZNVER1_M32: #define __SSE__ 1
// CHECK_ZNVER1_M32: #define __SSSE3__ 1
// CHECK_ZNVER1_M32: #define __XSAVEC__ 1
// CHECK_ZNVER1_M32: #define __XSAVEOPT__ 1
// CHECK_ZNVER1_M32: #define __XSAVES__ 1
// CHECK_ZNVER1_M32: #define __XSAVE__ 1
// CHECK_ZNVER1_M32: #define __i386 1
// CHECK_ZNVER1_M32: #define __i386__ 1
// CHECK_ZNVER1_M32: #define __tune_znver1__ 1
// CHECK_ZNVER1_M32: #define __znver1 1
// CHECK_ZNVER1_M32: #define __znver1__ 1
// RUN: %clang -march=znver1 -m64 -E -dM %s -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_ZNVER1_M64
// CHECK_ZNVER1_M64-NOT: #define __3dNOW_A__ 1
// CHECK_ZNVER1_M64-NOT: #define __3dNOW__ 1
// CHECK_ZNVER1_M64: #define __ADX__ 1
// CHECK_ZNVER1_M64: #define __AES__ 1
// CHECK_ZNVER1_M64: #define __AVX2__ 1
// CHECK_ZNVER1_M64: #define __AVX__ 1
// CHECK_ZNVER1_M64: #define __BMI2__ 1
// CHECK_ZNVER1_M64: #define __BMI__ 1
// CHECK_ZNVER1_M64: #define __CLFLUSHOPT__ 1
// CHECK_ZNVER1_M64: #define __CLZERO__ 1
// CHECK_ZNVER1_M64: #define __F16C__ 1
// CHECK_ZNVER1_M64: #define __FMA__ 1
// CHECK_ZNVER1_M64: #define __FSGSBASE__ 1
// CHECK_ZNVER1_M64: #define __LZCNT__ 1
// CHECK_ZNVER1_M64: #define __MMX__ 1
// CHECK_ZNVER1_M64: #define __PCLMUL__ 1
// CHECK_ZNVER1_M64: #define __POPCNT__ 1
// CHECK_ZNVER1_M64: #define __PRFCHW__ 1
// CHECK_ZNVER1_M64: #define __RDRND__ 1
// CHECK_ZNVER1_M64: #define __RDSEED__ 1
// CHECK_ZNVER1_M64: #define __SHA__ 1
// CHECK_ZNVER1_M64: #define __SSE2_MATH__ 1
// CHECK_ZNVER1_M64: #define __SSE2__ 1
// CHECK_ZNVER1_M64: #define __SSE3__ 1
// CHECK_ZNVER1_M64: #define __SSE4A__ 1
// CHECK_ZNVER1_M64: #define __SSE4_1__ 1
// CHECK_ZNVER1_M64: #define __SSE4_2__ 1
// CHECK_ZNVER1_M64: #define __SSE_MATH__ 1
// CHECK_ZNVER1_M64: #define __SSE__ 1
// CHECK_ZNVER1_M64: #define __SSSE3__ 1
// CHECK_ZNVER1_M64: #define __XSAVEC__ 1
// CHECK_ZNVER1_M64: #define __XSAVEOPT__ 1
// CHECK_ZNVER1_M64: #define __XSAVES__ 1
// CHECK_ZNVER1_M64: #define __XSAVE__ 1
// CHECK_ZNVER1_M64: #define __amd64 1
// CHECK_ZNVER1_M64: #define __amd64__ 1
// CHECK_ZNVER1_M64: #define __tune_znver1__ 1
// CHECK_ZNVER1_M64: #define __x86_64 1
// CHECK_ZNVER1_M64: #define __x86_64__ 1
// CHECK_ZNVER1_M64: #define __znver1 1
// CHECK_ZNVER1_M64: #define __znver1__ 1
//
// End X86/GCC/Linux tests ------------------

// Begin PPC/GCC/Linux tests ----------------
// Check that VSX also turns on altivec.
// RUN: %clang -mvsx -E -dM %s -o - 2>&1 \
// RUN:     -target powerpc-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PPC_VSX_M32
//
// CHECK_PPC_VSX_M32: #define __ALTIVEC__ 1
// CHECK_PPC_VSX_M32: #define __VSX__ 1
//
// RUN: %clang -mvsx -E -dM %s -o - 2>&1 \
// RUN:     -target powerpc64-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PPC_VSX_M64
//
// CHECK_PPC_VSX_M64: #define __VSX__ 1
//
// RUN: %clang -mpower8-vector -E -dM %s -o - 2>&1 \
// RUN:     -target powerpc64-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PPC_POWER8_VECTOR_M64
//
// CHECK_PPC_POWER8_VECTOR_M64: #define __POWER8_VECTOR__ 1
//
// RUN: %clang -mpower9-vector -E -dM %s -o - 2>&1 \
// RUN:     -target powerpc64-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PPC_POWER9_VECTOR_M64
//
// CHECK_PPC_POWER9_VECTOR_M64: #define __POWER9_VECTOR__ 1
//
// RUN: %clang -mcrypto -E -dM %s -o - 2>&1 \
// RUN:     -target powerpc64-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PPC_CRYPTO_M64
//
// CHECK_PPC_CRYPTO_M64: #define __CRYPTO__ 1

// HTM is available on power8 or later which includes all of powerpc64le as an
// ABI choice. Test that, the cpus, and the option.
// RUN: %clang -mhtm -E -dM %s -o - 2>&1 \
// RUN:     -target powerpc64-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PPC_HTM
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target powerpc64le-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PPC_HTM
// RUN: %clang -mcpu=pwr8 -E -dM %s -o - 2>&1 \
// RUN:     -target powerpc64-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PPC_HTM
// RUN: %clang -mcpu=pwr9 -E -dM %s -o - 2>&1 \
// RUN:     -target powerpc64-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PPC_HTM
//
// CHECK_PPC_HTM: #define __HTM__ 1

//
// RUN: %clang -mcpu=ppc64 -E -dM %s -o - 2>&1 \
// RUN:     -target powerpc64-unknown-unknown \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PPC_GCC_ATOMICS
// RUN: %clang -mcpu=pwr8 -E -dM %s -o - 2>&1 \
// RUN:     -target powerpc64-unknown-unknown \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PPC_GCC_ATOMICS
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target powerpc64le-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_PPC_GCC_ATOMICS
//
// CHECK_PPC_GCC_ATOMICS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK_PPC_GCC_ATOMICS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK_PPC_GCC_ATOMICS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK_PPC_GCC_ATOMICS: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
//
// End PPC/GCC/Linux tests ------------------

// Begin Sparc/GCC/Linux tests ----------------
//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target sparc-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SPARC
// RUN: %clang -mcpu=v9 -E -dM %s -o - 2>&1 \
// RUN:     -target sparc-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SPARC-V9
//
// CHECK_SPARC: #define __BIG_ENDIAN__ 1
// CHECK_SPARC: #define __sparc 1
// CHECK_SPARC: #define __sparc__ 1
// CHECK_SPARC-NOT: #define __sparcv9 1
// CHECK_SPARC-NOT: #define __sparcv9__ 1
// CHECK_SPARC: #define __sparcv8 1
// CHECK_SPARC-NOT: #define __sparcv9 1
// CHECK_SPARC-NOT: #define __sparcv9__ 1

// CHECK_SPARC-V9-NOT: #define __sparcv8 1
// CHECK_SPARC-V9: #define __sparc_v9__ 1
// CHECK_SPARC-V9: #define __sparcv9 1
// CHECK_SPARC-V9-NOT: #define __sparcv8 1

//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target sparcel-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SPARCEL
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=myriad2 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-2 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=myriad2.1 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-1 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=myriad2.2 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-2 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=myriad2.3 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-3 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=ma2100 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-1 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=ma2150 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-2 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=ma2155 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-2 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=ma2450 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-2 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=ma2455 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-2 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=ma2x5x 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-2 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=ma2080 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-3 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=ma2085 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-3 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=ma2480 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-3 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=ma2485 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-3 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// RUN: %clang -E -dM %s -o - -target sparcel-myriad -mcpu=ma2x8x 2>&1 \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_MYRIAD2-3 \
// RUN:     -check-prefix=CHECK_SPARCEL -check-prefix=CHECK_MYRIAD2
// CHECK_SPARCEL: #define __LITTLE_ENDIAN__ 1
// CHECK_MYRIAD2: #define __leon__ 1
// CHECK_MYRIAD2-1: #define __myriad2 1
// CHECK_MYRIAD2-1: #define __myriad2__ 1
// CHECK_MYRIAD2-2: #define __myriad2 2
// CHECK_MYRIAD2-2: #define __myriad2__ 2
// CHECK_MYRIAD2-3: #define __myriad2 3
// CHECK_MYRIAD2-3: #define __myriad2__ 3
// CHECK_SPARCEL: #define __sparc 1
// CHECK_SPARCEL: #define __sparc__ 1
// CHECK_MYRIAD2: #define __sparc_v8__ 1
// CHECK_SPARCEL: #define __sparcv8 1
//
// RUN: %clang -E -dM %s -o - 2>&1 \
// RUN:     -target sparcv9-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SPARCV9
//
// CHECK_SPARCV9: #define __BIG_ENDIAN__ 1
// CHECK_SPARCV9: #define __sparc 1
// CHECK_SPARCV9: #define __sparc64__ 1
// CHECK_SPARCV9: #define __sparc__ 1
// CHECK_SPARCV9: #define __sparc_v9__ 1
// CHECK_SPARCV9: #define __sparcv9 1
// CHECK_SPARCV9: #define __sparcv9__ 1

// Begin SystemZ/GCC/Linux tests ----------------
//
// RUN: %clang -march=arch8 -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_ARCH8
// RUN: %clang -march=z10 -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_ARCH8
//
// CHECK_SYSTEMZ_ARCH8: #define __ARCH__ 8
// CHECK_SYSTEMZ_ARCH8: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK_SYSTEMZ_ARCH8: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK_SYSTEMZ_ARCH8: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK_SYSTEMZ_ARCH8: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
// CHECK_SYSTEMZ_ARCH8: #define __LONG_DOUBLE_128__ 1
// CHECK_SYSTEMZ_ARCH8: #define __s390__ 1
// CHECK_SYSTEMZ_ARCH8: #define __s390x__ 1
// CHECK_SYSTEMZ_ARCH8: #define __zarch__ 1
//
// RUN: %clang -march=arch9 -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_ARCH9
// RUN: %clang -march=z196 -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_ARCH9
//
// CHECK_SYSTEMZ_ARCH9: #define __ARCH__ 9
// CHECK_SYSTEMZ_ARCH9: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK_SYSTEMZ_ARCH9: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK_SYSTEMZ_ARCH9: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK_SYSTEMZ_ARCH9: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
// CHECK_SYSTEMZ_ARCH9: #define __LONG_DOUBLE_128__ 1
// CHECK_SYSTEMZ_ARCH9: #define __s390__ 1
// CHECK_SYSTEMZ_ARCH9: #define __s390x__ 1
// CHECK_SYSTEMZ_ARCH9: #define __zarch__ 1
//
// RUN: %clang -march=arch10 -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_ARCH10
// RUN: %clang -march=zEC12 -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_ARCH10
//
// CHECK_SYSTEMZ_ARCH10: #define __ARCH__ 10
// CHECK_SYSTEMZ_ARCH10: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK_SYSTEMZ_ARCH10: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK_SYSTEMZ_ARCH10: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK_SYSTEMZ_ARCH10: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
// CHECK_SYSTEMZ_ARCH10: #define __HTM__ 1
// CHECK_SYSTEMZ_ARCH10: #define __LONG_DOUBLE_128__ 1
// CHECK_SYSTEMZ_ARCH10: #define __s390__ 1
// CHECK_SYSTEMZ_ARCH10: #define __s390x__ 1
// CHECK_SYSTEMZ_ARCH10: #define __zarch__ 1
//
// RUN: %clang -march=arch11 -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_ARCH11
// RUN: %clang -march=z13 -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_ARCH11
//
// CHECK_SYSTEMZ_ARCH11: #define __ARCH__ 11
// CHECK_SYSTEMZ_ARCH11: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK_SYSTEMZ_ARCH11: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK_SYSTEMZ_ARCH11: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK_SYSTEMZ_ARCH11: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
// CHECK_SYSTEMZ_ARCH11: #define __HTM__ 1
// CHECK_SYSTEMZ_ARCH11: #define __LONG_DOUBLE_128__ 1
// CHECK_SYSTEMZ_ARCH11: #define __VX__ 1
// CHECK_SYSTEMZ_ARCH11: #define __s390__ 1
// CHECK_SYSTEMZ_ARCH11: #define __s390x__ 1
// CHECK_SYSTEMZ_ARCH11: #define __zarch__ 1
//
// RUN: %clang -march=arch12 -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_ARCH12
// RUN: %clang -march=z14 -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_ARCH12
//
// CHECK_SYSTEMZ_ARCH12: #define __ARCH__ 12
// CHECK_SYSTEMZ_ARCH12: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK_SYSTEMZ_ARCH12: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK_SYSTEMZ_ARCH12: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK_SYSTEMZ_ARCH12: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
// CHECK_SYSTEMZ_ARCH12: #define __HTM__ 1
// CHECK_SYSTEMZ_ARCH12: #define __LONG_DOUBLE_128__ 1
// CHECK_SYSTEMZ_ARCH12: #define __VX__ 1
// CHECK_SYSTEMZ_ARCH12: #define __s390__ 1
// CHECK_SYSTEMZ_ARCH12: #define __s390x__ 1
// CHECK_SYSTEMZ_ARCH12: #define __zarch__ 1
//
// RUN: %clang -mhtm -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_HTM
//
// CHECK_SYSTEMZ_HTM: #define __HTM__ 1
//
// RUN: %clang -mvx -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_VX
//
// CHECK_SYSTEMZ_VX: #define __VX__ 1
//
// RUN: %clang -fzvector -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_ZVECTOR
// RUN: %clang -mzvector -E -dM %s -o - 2>&1 \
// RUN:     -target s390x-unknown-linux \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_SYSTEMZ_ZVECTOR
//
// CHECK_SYSTEMZ_ZVECTOR: #define __VEC__ 10302

// Begin amdgcn tests ----------------
//
// RUN: %clang -march=amdgcn -E -dM %s -o - 2>&1 \
// RUN:     -target amdgcn-unknown-unknown \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_AMDGCN
// CHECK_AMDGCN: #define __AMDGCN__ 1
// CHECK_AMDGCN: #define __HAS_FMAF__ 1
// CHECK_AMDGCN: #define __HAS_FP64__ 1
// CHECK_AMDGCN: #define __HAS_LDEXPF__ 1

// Begin r600 tests ----------------
//
// RUN: %clang -march=amdgcn -E -dM %s -o - 2>&1 \
// RUN:     -target r600-unknown-unknown \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_R600
// CHECK_R600: #define __R600__ 1
// CHECK_R600-NOT: #define __HAS_FMAF__ 1

// RUN: %clang -march=amdgcn -mcpu=cypress -E -dM %s -o - 2>&1 \
// RUN:     -target r600-unknown-unknown \
// RUN:   | FileCheck -match-full-lines %s -check-prefix=CHECK_R600_FP64
// CHECK_R600_FP64-DAG: #define __R600__ 1
// CHECK_R600_FP64-DAG: #define __HAS_FMAF__ 1
