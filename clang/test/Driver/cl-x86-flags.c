// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root
// REQUIRES: x86-registered-target

// We support -m32 and -m64.  We support all x86 CPU feature flags in gcc's -m
// flag space.
// RUN: %clang_cl /Zs /WX -m32 -m64 -msse3 -msse4.1 -mavx -mno-avx \
// RUN:     --target=i386-pc-win32 -### -- 2>&1 %s | FileCheck -check-prefix=MFLAGS %s
// MFLAGS-NOT: argument unused during compilation

// -arch:IA32 is no-op.
// RUN: %clang_cl -m32 -arch:IA32 --target=i386 -### -- 2>&1 %s | FileCheck -check-prefix=IA32 %s
// IA32-NOT: argument unused during compilation
// IA32-NOT: -target-feature

// RUN: %clang_cl -m32 -arch:ia32 --target=i386 -### -- 2>&1 %s | FileCheck -check-prefix=ia32 %s
// ia32: argument unused during compilation
// ia32-NOT: -target-feature

// RUN: %clang_cl -m64 -arch:IA32 --target=x86_64 -### -- 2>&1 %s | FileCheck -check-prefix=IA3264 %s
// IA3264: argument unused during compilation
// IA3264-NOT: -target-feature

// RUN: %clang_cl -m32 -arch:SSE --target=i386 -### -- 2>&1 %s | FileCheck -check-prefix=SSE %s
// SSE: -target-feature
// SSE: +sse
// SSE-NOT: argument unused during compilation

// RUN: %clang_cl -m32 -arch:sse --target=i386 -### -- 2>&1 %s | FileCheck -check-prefix=sse %s
// sse: argument unused during compilation
// sse-NOT: -target-feature

// RUN: %clang_cl -m32 -arch:SSE2 --target=i386 -### -- 2>&1 %s | FileCheck -check-prefix=SSE2 %s
// SSE2: -target-feature
// SSE2: +sse2
// SSE2-NOT: argument unused during compilation

// RUN: %clang_cl -m32 -arch:sse2 --target=i386 -### -- 2>&1 %s | FileCheck -check-prefix=sse %s
// sse2: argument unused during compilation
// sse2-NOT: -target-feature

// RUN: %clang_cl -m64 -arch:SSE --target=x86_64 -### -- 2>&1 %s | FileCheck -check-prefix=SSE64 %s
// SSE64: argument unused during compilation
// SSE64-NOT: -target-feature

// RUN: %clang_cl -m64 -arch:SSE2 --target=x86_64 -### -- 2>&1 %s | FileCheck -check-prefix=SSE264 %s
// SSE264: argument unused during compilation
// SSE264-NOT: -target-feature

// RUN: %clang_cl -m32 -arch:AVX --target=i386 -### -- 2>&1 %s | FileCheck -check-prefix=AVX %s
// AVX: -target-feature
// AVX: +avx

// RUN: %clang_cl -m32 -arch:avx --target=i386 -### -- 2>&1 %s | FileCheck -check-prefix=avx %s
// avx: argument unused during compilation
// avx-NOT: -target-feature

// RUN: %clang_cl -m32 -arch:AVX2 --target=i386 -### -- 2>&1 %s | FileCheck -check-prefix=AVX2 %s
// AVX2: -target-feature
// AVX2: +avx2

// RUN: %clang_cl -m32 -arch:avx2 --target=i386 -### -- 2>&1 %s | FileCheck -check-prefix=avx2 %s
// avx2: argument unused during compilation
// avx2-NOT: -target-feature

// RUN: %clang_cl -m64 -arch:AVX --target=x86_64 -### -- 2>&1 %s | FileCheck -check-prefix=AVX64 %s
// AVX64: -target-feature
// AVX64: +avx

// RUN: %clang_cl -m64 -arch:avx --target=x86_64 -### -- 2>&1 %s | FileCheck -check-prefix=avx64 %s
// avx64: argument unused during compilation
// avx64-NOT: -target-feature

// RUN: %clang_cl -m64 -arch:AVX2 --target=x86_64 -### -- 2>&1 %s | FileCheck -check-prefix=AVX264 %s
// AVX264: -target-feature
// AVX264: +avx2

// RUN: %clang_cl -m64 -arch:avx2 --target=x86_64 -### -- 2>&1 %s | FileCheck -check-prefix=avx264 %s
// avx264: argument unused during compilation
// avx264-NOT: -target-feature

void f() {
}
