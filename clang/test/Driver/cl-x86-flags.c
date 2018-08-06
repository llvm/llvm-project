// REQUIRES: x86-registered-target

// expected-no-diagnostics

// We support -m32 and -m64.  We support all x86 CPU feature flags in gcc's -m
// flag space.
// RUN: %clang_cl /Zs /WX -m32 -m64 -msse3 -msse4.1 -mavx -mno-avx \
// RUN:     --target=i386-pc-win32 -### -- 2>&1 %s | FileCheck -check-prefix=MFLAGS %s
// MFLAGS-NOT: argument unused during compilation

// RUN: %clang_cl -m32 -arch:IA32 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_IA32 -- %s
#if defined(TEST_32_ARCH_IA32)
#if _M_IX86_FP || __AVX__ || __AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// arch: args are case-sensitive.
// RUN: %clang_cl -m32 -arch:ia32 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=ia32 %s
// ia32: argument unused during compilation

// RUN: %clang_cl -m64 -arch:IA32 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=IA3264 %s
// IA3264: argument unused during compilation

// RUN: %clang_cl -m32 -arch:SSE --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_SSE -- %s
#if defined(TEST_32_ARCH_SSE)
#if _M_IX86_FP != 1 || __AVX__ || __AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:sse --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=sse %s
// sse: argument unused during compilation

// RUN: %clang_cl -m32 -arch:SSE2 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_SSE2 -- %s
#if defined(TEST_32_ARCH_SSE2)
#if _M_IX86_FP != 2 || __AVX__ || __AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:sse2 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=sse %s
// sse2: argument unused during compilation

// RUN: %clang_cl -m64 -arch:SSE --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=SSE64 %s
// SSE64: argument unused during compilation

// RUN: %clang_cl -m64 -arch:SSE2 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=SSE264 %s
// SSE264: argument unused during compilation

// RUN: %clang_cl -m32 -arch:AVX --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_AVX -- %s
#if defined(TEST_32_ARCH_AVX)
#if _M_IX86_FP != 2 || !__AVX__ || __AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:avx --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx %s
// avx: argument unused during compilation

// RUN: %clang_cl -m32 -arch:AVX2 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_AVX2 -- %s
#if defined(TEST_32_ARCH_AVX2)
#if _M_IX86_FP != 2 || !__AVX__ || !__AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:avx2 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx2 %s
// avx2: argument unused during compilation

// RUN: %clang_cl -m32 -arch:AVX512F --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_AVX512F -- %s
#if defined(TEST_32_ARCH_AVX512F)
#if _M_IX86_FP != 2 || !__AVX__ || !__AVX2__ || !__AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:avx512f --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx512f %s
// avx512f: argument unused during compilation

// RUN: %clang_cl -m32 -arch:AVX512 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_AVX512 -- %s
#if defined(TEST_32_ARCH_AVX512)
#if _M_IX86_FP != 2 || !__AVX__ || !__AVX2__ || !__AVX512F__  || !__AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:avx512 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx512 %s
// avx512: argument unused during compilation

// RUN: %clang_cl -m64 -arch:AVX --target=x86_64-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_64_ARCH_AVX -- %s
#if defined(TEST_64_ARCH_AVX)
#if _M_IX86_FP || !__AVX__ || __AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m64 -arch:avx --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx64 %s
// avx64: argument unused during compilation

// RUN: %clang_cl -m64 -arch:AVX2 --target=x86_64-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_64_ARCH_AVX2 -- %s
#if defined(TEST_64_ARCH_AVX2)
#if _M_IX86_FP || !__AVX__ || !__AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m64 -arch:avx2 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx264 %s
// avx264: argument unused during compilation

// RUN: %clang_cl -m64 -arch:AVX512F --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_64_ARCH_AVX512F -- %s
#if defined(TEST_64_ARCH_AVX512F)
#if _M_IX86_FP || !__AVX__ || !__AVX2__ || !__AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m64 -arch:avx512f --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx512f64 %s
// avx512f64: argument unused during compilation

// RUN: %clang_cl -m64 -arch:AVX512 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_64_ARCH_AVX512 -- %s
#if defined(TEST_64_ARCH_AVX512)
#if _M_IX86_FP || !__AVX__ || !__AVX2__ || !__AVX512F__  || !__AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m64 -arch:avx512 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx51264 %s
// avx51264: argument unused during compilation

void f() {
}
