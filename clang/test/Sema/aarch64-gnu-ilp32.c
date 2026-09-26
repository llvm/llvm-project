// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu_ilp32 -std=c11 -fsyntax-only %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu_ilp32 -x c++ -std=c++17 -fsyntax-only %s
// RUN: %clang_cc1 -triple aarch64_be-unknown-linux-gnu_ilp32 -std=c11 -fsyntax-only %s

#if !defined(__ILP32__) || defined(_LP64) || defined(__LP64__)
#error AArch64 GNU ILP32 must define __ILP32__ and not LP64 macros
#endif

#ifdef __cplusplus
#define CHECK static_assert
#else
#define CHECK _Static_assert
#endif

CHECK(sizeof(int) == 4, "int must be 32 bits");
CHECK(sizeof(long) == 4, "long must be 32 bits");
CHECK(sizeof(void *) == 4, "pointers must be 32 bits");
CHECK(__alignof__(void *) == 4, "pointers must have 32-bit alignment");
CHECK(sizeof(long long) == 8, "long long must be 64 bits");
CHECK(sizeof(__INT64_TYPE__) == 8, "int64_t's underlying type must be 64 bits");
CHECK(sizeof(__INTMAX_TYPE__) == 8, "intmax_t's underlying type must be 64 bits");
CHECK(sizeof(__SIZE_TYPE__) == 4, "size_t's underlying type must be 32 bits");
CHECK(sizeof(__PTRDIFF_TYPE__) == 4, "ptrdiff_t's underlying type must be 32 bits");
CHECK(sizeof(__INTPTR_TYPE__) == 4, "intptr_t's underlying type must be 32 bits");
CHECK(__SIZEOF_POINTER__ == 4, "the pointer size macro must be 4");
CHECK(__SIZEOF_LONG__ == 4, "the long size macro must be 4");
