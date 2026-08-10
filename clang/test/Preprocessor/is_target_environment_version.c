// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-windows-msvc18.0.0 -verify %s
// expected-no-diagnostics

#if !__is_target_environment(msvc)
  #error "mismatching environment"
#endif
