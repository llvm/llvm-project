// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macos -DMAC -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-ios -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-tvos -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-watchos -verify %s
// expected-no-diagnostics

#if !__is_target_os(darwin)
  #error "mismatching os"
#endif

// macOS matches both macOS and macOSX.
#ifdef MAC

#if !__is_target_os(macos)
  #error "mismatching os"
#endif

#if !__is_target_os(macosx)
  #error "mismatching os"
#endif

#if __is_target_os(ios)
  #error "mismatching os"
#endif

#endif
