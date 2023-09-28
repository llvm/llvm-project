// RUN: %clang_cc1 %s -E -CC -verify
// RUN: %clang_cc1 -x c %s -E -CC -verify

#if defined(__cplusplus)
#if !defined(__cpp_pp_embed) || __cpp_pp_embed != 202403L
#error 1
#endif
#endif

#if !defined(__has_embed)
#error 2
#endif
// expected-no-diagnostics
