/// Tests that SPIR-V device targets adapt pointer and integer type sizes
/// from the host target via -aux-triple.

// RUN: %clang_cc1 -fsycl-is-device \
// RUN:   -triple spirv64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsyntax-only -verify=linux64 %s
// RUN: %clang_cc1 -fsycl-is-device \
// RUN:   -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsyntax-only -verify=win64 %s
// RUN: %clang_cc1 -fsycl-is-device \
// RUN:   -triple spirv32-unknown-unknown -aux-triple i386-unknown-linux-gnu \
// RUN:   -fsyntax-only -verify=linux32 %s
// RUN: %clang_cc1 -fsycl-is-device \
// RUN:   -triple spirv32-unknown-unknown -aux-triple i386-pc-windows-msvc \
// RUN:   -fsyntax-only -verify=win32 %s
// RUN: %clang_cc1 -fsycl-is-device \
// RUN:   -triple spirv64-unknown-unknown \
// RUN:   -fsyntax-only -verify=nohost64 %s
// RUN: %clang_cc1 -fsycl-is-device \
// RUN:   -triple spirv32-unknown-unknown \
// RUN:   -fsyntax-only -verify=nohost32 %s

// linux64-no-diagnostics
// win64-no-diagnostics
// linux32-no-diagnostics
// win32-no-diagnostics
// nohost64-no-diagnostics
// nohost32-no-diagnostics

typedef __SIZE_TYPE__ size_t_type;
typedef __PTRDIFF_TYPE__ ptrdiff_t_type;
typedef __INTPTR_TYPE__ intptr_t_type;

// --- SPIRV64 + Linux x86_64 (LP64): long=8, pointer=8 ---
#if __SPIRV64__ && defined(__linux__) && defined(__x86_64__)
static_assert(sizeof(void *) == 8, "pointer should be 64-bit");
static_assert(sizeof(long) == 8, "long should be 64-bit with Linux LP64");
static_assert(sizeof(size_t_type) == 8, "size_t must be 64-bit");
static_assert(sizeof(ptrdiff_t_type) == 8, "ptrdiff_t must be 64-bit");
static_assert(sizeof(intptr_t_type) == 8, "intptr_t must be 64-bit");
#endif

// --- SPIRV64 + Windows x86_64 (LLP64): long=4, pointer=8 ---
#if __SPIRV64__ && defined(_WIN64)
static_assert(sizeof(void *) == 8, "pointer should be 64-bit");
static_assert(sizeof(long) == 4, "long should be 32-bit with Windows LLP64");
static_assert(sizeof(size_t_type) == 8, "size_t must be 64-bit");
static_assert(sizeof(ptrdiff_t_type) == 8, "ptrdiff_t must be 64-bit");
static_assert(sizeof(intptr_t_type) == 8, "intptr_t must be 64-bit");
#endif

// --- SPIRV32 + Linux i386 (ILP32): long=4, pointer=4 ---
#if __SPIRV32__ && defined(__linux__) && defined(__i386__)
static_assert(sizeof(void *) == 4, "pointer should be 32-bit");
static_assert(sizeof(long) == 4, "long should be 32-bit with ILP32");
static_assert(sizeof(size_t_type) == 4, "size_t must be 32-bit");
static_assert(sizeof(ptrdiff_t_type) == 4, "ptrdiff_t must be 32-bit");
static_assert(sizeof(intptr_t_type) == 4, "intptr_t must be 32-bit");
#endif

// --- SPIRV32 + Windows i386 (ILP32): long=4, pointer=4 ---
#if __SPIRV32__ && defined(_WIN32) && !defined(_WIN64)
static_assert(sizeof(void *) == 4, "pointer should be 32-bit");
static_assert(sizeof(long) == 4, "long should be 32-bit on Win32");
static_assert(sizeof(size_t_type) == 4, "size_t must be 32-bit");
static_assert(sizeof(ptrdiff_t_type) == 4, "ptrdiff_t must be 32-bit");
static_assert(sizeof(intptr_t_type) == 4, "intptr_t must be 32-bit");
#endif

// --- SPIRV64 no host (defaults match LP64) ---
#if __SPIRV64__ && !defined(__linux__) && !defined(_WIN64)
static_assert(sizeof(void *) == 8, "pointer should be 64-bit");
static_assert(sizeof(long) == 8, "long should be 64-bit with default LP64");
static_assert(sizeof(size_t_type) == 8, "size_t must be 64-bit");
static_assert(sizeof(ptrdiff_t_type) == 8, "ptrdiff_t must be 64-bit");
static_assert(sizeof(intptr_t_type) == 8, "intptr_t must be 64-bit");
#endif

// --- SPIRV32 no host (pointer=4, but long stays at base default=8) ---
#if __SPIRV32__ && !defined(__linux__) && !defined(_WIN32)
static_assert(sizeof(void *) == 4, "pointer should be 32-bit");
static_assert(sizeof(long) == 8, "long defaults to 64-bit without a host");
static_assert(sizeof(size_t_type) == 4, "size_t must be 32-bit");
static_assert(sizeof(ptrdiff_t_type) == 4, "ptrdiff_t must be 32-bit");
static_assert(sizeof(intptr_t_type) == 4, "intptr_t must be 32-bit");
#endif
