// RUN: %clang -target i386-unknown-unknown -march=core2 -msse4 -x c -E -dM -o %t %s
// RUN: grep '#define __SSE2_MATH__ 1' %t
// RUN: grep '#define __SSE2__ 1' %t
// RUN: grep '#define __SSE3__ 1' %t
// RUN: grep '#define __SSE4_1__ 1' %t
// RUN: grep '#define __SSE4_2__ 1' %t
// RUN: grep '#define __SSE_MATH__ 1' %t
// RUN: grep '#define __SSE__ 1' %t
// RUN: grep '#define __SSSE3__ 1' %t

// RUN: %clang -target i386-unknown-unknown -march=core2 -msse4 -mno-sse2 -x c -E -dM -o %t %s
// RUN: grep '#define __SSE2_MATH__ 1' %t | count 0
// RUN: grep '#define __SSE2__ 1' %t | count 0
// RUN: grep '#define __SSE3__ 1' %t | count 0
// RUN: grep '#define __SSE4_1__ 1' %t | count 0
// RUN: grep '#define __SSE4_2__ 1' %t | count 0
// RUN: grep '#define __SSE_MATH__ 1' %t
// RUN: grep '#define __SSE__ 1' %t
// RUN: grep '#define __SSSE3__ 1' %t | count 0

// RUN: %clang -target i386-unknown-unknown -march=pentium-m -x c -E -dM -o %t %s
// RUN: grep '#define __SSE2_MATH__ 1' %t
// RUN: grep '#define __SSE2__ 1' %t
// RUN: grep '#define __SSE3__ 1' %t | count 0
// RUN: grep '#define __SSE4_1__ 1' %t | count 0
// RUN: grep '#define __SSE4_2__ 1' %t | count 0
// RUN: grep '#define __SSE_MATH__ 1' %t
// RUN: grep '#define __SSE__ 1' %t
// RUN: grep '#define __SSSE3__ 1' %t | count 0



