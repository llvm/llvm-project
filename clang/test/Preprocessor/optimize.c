// RUN: %clang_cc1 -Eonly %s -DOPT_O2 -O2 -verify
#ifdef OPT_O2
  // expected-no-diagnostics
  #ifndef __OPTIMIZE__
    #error "__OPTIMIZE__ not defined"
  #endif
  #ifdef __OPTIMIZE_SIZE__
    #error "__OPTIMIZE_SIZE__ defined"
  #endif
#endif

// RUN: %clang_cc1 -Eonly %s -DOPT_O0 -O0 -verify
#ifdef OPT_O0
  // expected-no-diagnostics
  #ifdef __OPTIMIZE__
    #error "__OPTIMIZE__ defined"
  #endif
  #ifdef __OPTIMIZE_SIZE__
    #error "__OPTIMIZE_SIZE__ defined"
  #endif
#endif

// RUN: %clang_cc1 -Eonly %s -DOPT_OS -Os -verify
#ifdef OPT_OS
  // expected-no-diagnostics
  #ifndef __OPTIMIZE__
    #error "__OPTIMIZE__ not defined"
  #endif
  #ifndef __OPTIMIZE_SIZE__
    #error "__OPTIMIZE_SIZE__ not defined"
  #endif
#endif
