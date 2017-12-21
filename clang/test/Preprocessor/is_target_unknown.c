// RUN: %clang_cc1 -fsyntax-only -triple i686-unknown-unknown -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple i686-- -verify %s
// expected-no-diagnostics

#if __is_target_arch(unknown)
  #error "mismatching arch"
#endif

// Unknown vendor is allowed.
#if !__is_target_vendor(unknown)
  #error "mismatching vendor"
#endif

// Unknown OS is allowed.
#if !__is_target_os(unknown)
  #error "mismatching OS"
#endif

// Unknown environment is allowed.
#if !__is_target_environment(unknown)
  #error "mismatching environment"
#endif
