// RUN: %clang_cc1 -fsyntax-only -triple arm64-apple-ios11 -verify %s
// expected-no-diagnostics

#if !__is_target_arch(arm64) || !__is_target_arch(aarch64)
  #error "mismatching arch"
#endif

#if __is_target_arch(aarch64_be)
  #error "mismatching arch"
#endif
