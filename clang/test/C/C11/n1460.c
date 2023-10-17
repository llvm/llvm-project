// RUN: %clang_cc1 -verify -Wno-vla %s

/* WG14 N1460: Yes
 * Subsetting the standard
 */

// If we claim to not support the feature then we expect diagnostics when
// using that feature. Otherwise, we expect no diagnostics.
#ifdef __STDC_NO_COMPLEX__
  // PS4/PS5 set this to indicate no <complex.h> but still support the
  // _Complex syntax.
  #ifdef __SCE__
    #define HAS_COMPLEX
  #else
    // We do not have any other targets which do not support complex, so we
    // don't expect to get into this block.
    #error "it's unexpected that we don't support complex"
  #endif
  float _Complex fc;
  double _Complex dc;
  long double _Complex ldc;
#else
  #define HAS_COMPLEX
  float _Complex fc;
  double _Complex dc;
  long double _Complex ldc;
#endif

#ifdef __STDC_NO_VLA__
  // We do not have any targets which do not support VLAs, so we don't expect
  // to get into this block.
  #error "it's unexpected that we don't support VLAs"

  void func(int n, int m[n]) {
    int array[n];
  }
#else
  #define HAS_VLA
  void func(int n, int m[n]) {
    int array[n];
  }
#endif

// NB: it's not possible to test for __STDC_NO_THREADS__ because that is
// specifically about whether <threads.h> exists and is supported, which is
// outside the control of the compiler. It does not cover use of thread_local.

#if defined(HAS_COMPLEX) && defined(HAS_VLA)
// If we support all these optional features, we don't expect any other
// diagnostics to have fired.

// expected-no-diagnostics
#endif

