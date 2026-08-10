// RUN: %clang -march=i386 -m32 -E -dM %s -msse -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck %s -check-prefix=SSE_AND_MMX
// RUN: %clang -march=i386 -m32 -E -dM %s -msse -mno-mmx -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck %s -check-prefix=SSE_NO_MMX
// RUN: %clang -march=i386 -m32 -E -dM %s -mno-mmx -msse -o - 2>&1 \
// RUN:     -target i386-unknown-linux \
// RUN:   | FileCheck %s -check-prefix=SSE_NO_MMX

// SSE_AND_MMX: #define __MMX__
// SSE_AND_MMX: #define __SSE__

// SSE_NO_MMX-NOT: __MMX__
// SSE_NO_MMX: __SSE__
// SSE_NO_MMX-NOT: __MMX__
