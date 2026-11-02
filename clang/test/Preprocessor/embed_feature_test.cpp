// RUN: %clang_cc1 %s -E -CC -verify
// RUN: %clang_cc1 -x c %s -E -CC -verify
// expected-no-diagnostics

#if !defined(__has_embed)
#error 1
#endif

#ifdef __cplusplus
#if __cpp_pp_embed != 202502L
#error 2
#endif
#else
#ifdef __cpp_pp_embed
#error 3
#endif
#endif
