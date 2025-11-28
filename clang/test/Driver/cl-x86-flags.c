// REQUIRES: x86-registered-target

// expected-no-diagnostics

// We support -m32 and -m64.  We support all x86 CPU feature flags in gcc's -m
// flag space.
// RUN: %clang_cl /Zs /WX -m32 -m64 -msse3 -msse4.1 -mavx -mno-avx \
// RUN:     --target=i386-pc-win32 -### -- 2>&1 %s | FileCheck -check-prefix=MFLAGS %s
// MFLAGS-NOT: invalid /arch: argument
//

// RUN: %clang_cl -m32 -arch:IA32 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_IA32 -- %s
#if defined(TEST_32_ARCH_IA32)
#if _M_IX86_FP || __AVX__ || __AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// arch: args are case-sensitive.
// RUN: %clang_cl -m32 -arch:ia32 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=ia32 %s
// ia32: invalid /arch: argument 'ia32'; for 32-bit expected one of AVX, AVX10.1, AVX10.2, AVX2, AVX512, AVX512F, IA32, SSE, SSE2

// RUN: %clang_cl -m64 -arch:IA32 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=IA3264 %s
// IA3264: invalid /arch: argument 'IA32'; for 64-bit expected one of AVX, AVX10.1, AVX10.2, AVX2, AVX512, AVX512F

// RUN: %clang_cl -m32 -arch:SSE --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_SSE -- %s
#if defined(TEST_32_ARCH_SSE)
#if _M_IX86_FP != 1 || __AVX__ || __AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:sse --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=sse %s
// sse: invalid /arch: argument

// RUN: %clang_cl -m32 -arch:SSE2 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_SSE2 -- %s
#if defined(TEST_32_ARCH_SSE2)
#if _M_IX86_FP != 2 || __AVX__ || __AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:sse2 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=sse %s
// sse2: invalid /arch: argument

// RUN: %clang_cl -m64 -arch:SSE --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=SSE64 %s
// SSE64: invalid /arch: argument 'SSE'; for 64-bit expected one of AVX, AVX10.1, AVX10.2, AVX2, AVX512, AVX512F

// RUN: %clang_cl -m64 -arch:SSE2 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=SSE264 %s
// SSE264: invalid /arch: argument

// RUN: %clang_cl -m32 -arch:AVX --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_AVX -- %s
#if defined(TEST_32_ARCH_AVX)
#if _M_IX86_FP != 2 || !__AVX__ || __AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:avx --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx %s
// avx: invalid /arch: argument

// RUN: %clang_cl -m32 -arch:AVX2 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_AVX2 -- %s
#if defined(TEST_32_ARCH_AVX2)
#if _M_IX86_FP != 2 || !__AVX__ || !__AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:avx2 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx2 %s
// avx2: invalid /arch: argument

// RUN: %clang_cl -m32 -arch:AVX512F --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_AVX512F -- %s
#if defined(TEST_32_ARCH_AVX512F)
#if _M_IX86_FP != 2 || !__AVX__ || !__AVX2__ || !__AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:avx512f --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx512f %s
// avx512f: invalid /arch: argument

// RUN: %clang_cl -m32 -arch:AVX512 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_AVX512 -- %s
// RUN: %clang_cl -m32 -arch:AVX10.1 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_AVX512 -- %s
// RUN: %clang_cl -m32 -arch:AVX10.2 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_AVX512 -- %s
#if defined(TEST_32_ARCH_AVX512)
#if _M_IX86_FP != 2 || !__AVX__ || !__AVX2__ || !__AVX512F__  || !__AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:avx512 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx512 %s
// avx512: invalid /arch: argument

// RUN: %clang_cl -m32 -arch:AVX10.1 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_AVX10_1_ADD -- %s
// RUN: %clang_cl -m32 -arch:AVX10.2 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_AVX10_1_ADD -- %s
#if defined(TEST_32_ARCH_AVX10_1_ADD)
#if !__AVX512VL__  || !__AVX512CD__ || !__AVX512DQ__ || !__AVX512VBMI__ || !__AVX512IFMA__ || !__AVX512VNNI__ ||\
    !__AVX512BF16__ || !__AVX512VPOPCNTDQ__ || !__AVX512VBMI2__ || !__AVX512VPOPCNTDQ__ || !__AVX512BITALG__ || !__AVX512FP16__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:avx10.1 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx10_1 %s
// avx10_1: invalid /arch: argument

// RUN: %clang_cl -m32 -arch:AVX10.2 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_32_ARCH_AVX10_2_ADD -- %s
#if defined(TEST_32_ARCH_AVX10_2_ADD)
#if !__AVX10_2__
#error fail
#endif
#endif

// RUN: %clang_cl -m32 -arch:avx10.2 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx10_2 %s
// avx10_2: invalid /arch: argument

// RUN: %clang_cl -m64 -arch:AVX --target=x86_64-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_64_ARCH_AVX -- %s
#if defined(TEST_64_ARCH_AVX)
#if _M_IX86_FP || !__AVX__ || __AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m64 -arch:avx --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx64 %s
// avx64: invalid /arch: argument

// RUN: %clang_cl -m64 -arch:AVX2 --target=x86_64-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_64_ARCH_AVX2 -- %s
#if defined(TEST_64_ARCH_AVX2)
#if _M_IX86_FP || !__AVX__ || !__AVX2__ || __AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m64 -arch:avx2 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx264 %s
// avx264: invalid /arch: argument

// RUN: %clang_cl -m64 -arch:AVX512F --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_64_ARCH_AVX512F -- %s
#if defined(TEST_64_ARCH_AVX512F)
#if _M_IX86_FP || !__AVX__ || !__AVX2__ || !__AVX512F__  || __AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m64 -arch:avx512f --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx512f64 %s
// avx512f64: invalid /arch: argument

// RUN: %clang_cl -m64 -arch:AVX512 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_64_ARCH_AVX512 -- %s
// RUN: %clang_cl -m64 -arch:AVX10.1 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_64_ARCH_AVX512 -- %s
// RUN: %clang_cl -m64 -arch:AVX10.2 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_64_ARCH_AVX512 -- %s
#if defined(TEST_64_ARCH_AVX512)
#if _M_IX86_FP || !__AVX__ || !__AVX2__ || !__AVX512F__  || !__AVX512BW__
#error fail
#endif
#endif

// RUN: %clang_cl -m64 -arch:avx512 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx51264 %s
// avx51264: invalid /arch: argument

// RUN: %clang_cl -m64 -arch:AVX10.1 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_64_ARCH_AVX10_1_ADD -- %s
// RUN: %clang_cl -m64 -arch:AVX10.2 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_64_ARCH_AVX10_1_ADD -- %s
#if defined(TEST_64_ARCH_AVX10_1_ADD)
#if !__AVX512VL__  || !__AVX512CD__ || !__AVX512DQ__ || !__AVX512VBMI__ || !__AVX512IFMA__ || !__AVX512VNNI__ ||\
    !__AVX512BF16__ || !__AVX512VPOPCNTDQ__ || !__AVX512VBMI2__ || !__AVX512VPOPCNTDQ__ || !__AVX512BITALG__ || !__AVX512FP16__
#error fail
#endif
#endif

// RUN: %clang_cl -m64 -arch:avx10.1 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx10_1_64 %s
// avx10_1_64: invalid /arch: argument

// RUN: %clang_cl -m64 -arch:AVX10.2 --target=i386-pc-windows /c /Fo%t.obj -Xclang -verify -DTEST_64_ARCH_AVX10_2_ADD -- %s
#if defined(TEST_64_ARCH_AVX10_2_ADD)
#if !__AVX10_2__
#error fail
#endif
#endif

// RUN: %clang_cl -m64 -arch:avx10.2 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx10_2_64 %s
// avx10_2_64: invalid /arch: argument

// RUN: %clang_cl -m64 -arch:AVX -tune:haswell --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=tune %s
// tune: "-target-cpu" "sandybridge"
// tune-SAME: "-tune-cpu" "haswell"

// RUN: %clang_cl -m64 -arch:AVX512 -vlen=512 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=vlen512 %s
// RUN: %clang_cl -m64 -arch:AVX10.1 -vlen=512 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=vlen512 %s
// RUN: %clang_cl -m64 -arch:AVX10.2 -vlen=512 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=vlen512 %s
// vlen512: "-mprefer-vector-width=512"

// RUN: %clang_cl -m64 -arch:AVX512 -vlen=256 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=vlen256 %s
// RUN: %clang_cl -m64 -arch:AVX10.1 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=vlen256 %s
// RUN: %clang_cl -m64 -arch:AVX10.2 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=vlen256 %s
// RUN: %clang_cl -m64 -arch:AVX10.1 -vlen=256 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=vlen256 %s
// RUN: %clang_cl -m64 -arch:AVX10.2 -vlen=256 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=vlen256 %s
// vlen256: "-mprefer-vector-width=256"

// RUN: %clang_cl -m64 -arch:AVX512 -vlen=512 -vlen --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=novlen %s
// novlen-NOT: -mprefer-vector-width

// RUN: %clang_cl -m64 -arch:AVX2 -vlen=512 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx2vlen512 %s
// avx2vlen512: invalid argument '/vlen=512' not allowed with '/arch:AVX2'

// RUN: %clang_cl -m64 -arch:AVX2 -vlen=256 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=avx2vlen256 %s
// avx2vlen256-NOT: invalid argument

// RUN: %clang_cl -m32 -arch:SSE2 -vlen=256 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=sse2vlen256 %s
// RUN: %clang_cl -m64 -vlen=256 --target=x86_64-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=sse2vlen256 %s
// sse2vlen256: invalid argument '/vlen=256' not allowed with '/arch:SSE2'

// RUN: %clang_cl -m32 -vlen=256 --target=i386-pc-windows -### -- 2>&1 %s | FileCheck -check-prefix=ia32vlen256 %s
// ia32vlen256: invalid argument '/vlen=256' not allowed with '/arch:IA32'

void f(void) {
}


// RUN: not %clang_cl -### --target=i386-pc-windows -mapx-features=ndd -- 2>&1 %s | FileCheck --check-prefix=NON-APX %s
// RUN: not %clang_cl -### --target=i386-pc-windows -mapxf -- 2>&1 %s | FileCheck --check-prefix=NON-APX %s
// RUN: %clang_cl -### --target=i386-pc-windows -mno-apxf -- 2>&1 %s > /dev/null
// NON-APX:      error: unsupported option '-mapx-features=|-mapxf' for target 'i386-pc-windows{{.*}}'
// NON-APX-NOT:  error: {{.*}} -mapx-features=

// RUN: %clang_cl --target=x86_64-pc-windows -mapxf -### -- 2>&1 %s | FileCheck -check-prefix=APXF %s
// RUN: %clang_cl --target=x86_64-pc-windows -mapxf -mno-apxf -### -- 2>&1 %s | FileCheck -check-prefix=NO-APXF %s
// RUN: %clang_cl --target=x86_64-pc-windows -mapx-features=egpr,push2pop2,ppx,ndd,ccmp,nf,cf,zu -### -- 2>&1 %s | FileCheck -check-prefix=APXALL %s
// APXF: "-target-feature" "+egpr" "-target-feature" "+push2pop2" "-target-feature" "+ppx" "-target-feature" "+ndd" "-target-feature" "+ccmp" "-target-feature" "+nf" "-target-feature" "+zu"
// NO-APXF: "-target-feature" "-egpr" "-target-feature" "-push2pop2" "-target-feature" "-ppx" "-target-feature" "-ndd" "-target-feature" "-ccmp" "-target-feature" "-nf" "-target-feature" "-zu"
// APXALL: "-target-feature" "+egpr" "-target-feature" "+push2pop2" "-target-feature" "+ppx" "-target-feature" "+ndd" "-target-feature" "+ccmp" "-target-feature" "+nf" "-target-feature" "+cf" "-target-feature" "+zu"
