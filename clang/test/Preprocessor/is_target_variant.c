// RUN: %clang_cc1 -fsyntax-only -triple arm64-apple-macos -DMAC -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple arm64-apple-ios13.1 -DIOS -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple arm64-apple-ios13.1-macabi -DCATALYST -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple arm64-apple-macos12 -darwin-target-variant-triple arm64-apple-ios-macabi -DZIPPERED -verify %s
// expected-no-diagnostics

#if !__has_builtin(__is_target_variant_os) || !__has_builtin(__is_target_variant_environment)
  #error "has builtin doesn't work"
#endif

#ifdef ZIPPERED

  // Target variant is a darwin.
  #if !__is_target_variant_os(darwin)
    #error "mismatching variant os"
  #endif

  // Target variant is not macOS...
  #if __is_target_variant_os(macos)
    #error "mismatching variant os"
  #endif

  // ...but iOS.
  #if !__is_target_variant_os(ios)
    #error "mismatching variant os"
  #endif

  // Zippered builds also set the target variant environment to macabi.
  // At the moment, only zippered builds set __is_target_variant_os(ios),
  // so checking __is_target_variant_environment() is currently redundant
  // with checking the former.
  #if !__is_target_variant_environment(macabi)
    #error "mismatching variant environment"
  #endif

#else

  // In non-zippered builds, even for catalyst, no target variant is set.
  // So these are all false.

  #if __is_target_variant_os(darwin)
    #error "mismatching variant os"
  #endif

  #if __is_target_variant_os(macos)
    #error "mismatching variant os"
  #endif

  #if __is_target_variant_os(ios)
    #error "mismatching variant os"
  #endif

  #if __is_target_variant_environment(macabi)
    #error "mismatching variant environment"
  #endif

#endif

// The target environment in zippered builds is _not_ macabi.
// The target environment is macabi only in catalyst builds.
#ifdef CATALYST
  #if !__is_target_environment(macabi)
    #error "mismatching environment"
  #endif
  #if !__is_target_os(ios)
    #error "mismatching os"
  #endif
#else
  #if __is_target_environment(macabi)
    #error "mismatching environment"
  #endif
#endif
