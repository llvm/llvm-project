/// Tests SPIRV64 with Windows MSVC aux-triple: LLP64 type sizes.
// RUN: %clang_cc1 -fsycl-is-device \
// RUN:   -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsyntax-only -verify %s

// expected-no-diagnostics

static_assert(sizeof(void *) == 8, "pointer should be 64-bit on spirv64");
static_assert(sizeof(long) == 4, "long should be 32-bit with Windows aux-triple");

typedef __SIZE_TYPE__ size_t_type;
static_assert(sizeof(size_t_type) == 8, "size_t must be 64-bit on Win64 SPIRV64");

typedef __PTRDIFF_TYPE__ ptrdiff_t_type;
static_assert(sizeof(ptrdiff_t_type) == 8, "ptrdiff_t must be 64-bit on Win64 SPIRV64");

typedef __INTPTR_TYPE__ intptr_t_type;
static_assert(sizeof(intptr_t_type) == 8, "intptr_t must be 64-bit on Win64 SPIRV64");
