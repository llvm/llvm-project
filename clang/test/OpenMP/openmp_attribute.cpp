// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fsyntax-only -verify -DSUPPORTED=1 %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fsyntax-only -verify -DSUPPORTED=1 -x c -std=c2x %s
// RUN: %clang_cc1 -fsyntax-only -verify -DSUPPORTED=0 %s
// RUN: %clang_cc1 -fsyntax-only -verify -DSUPPORTED=0 -x c -std=c2x %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fsyntax-only -verify -DSUPPORTED=1 %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fsyntax-only -verify -DSUPPORTED=1 -x c -std=c2x %s
// expected-no-diagnostics

#ifndef SUPPORTED
#error "Someone messed up a RUN line"
#endif

#ifdef __cplusplus
#if __has_cpp_attribute(omp::sequence) != SUPPORTED
#error "No idea what you're talking about"
#endif

#if __has_cpp_attribute(omp::directive) != SUPPORTED
#error "No idea what you're talking about"
#endif

#if __has_cpp_attribute(omp::totally_bogus)
#error "No idea what you're talking about"
#endif

#else // __cplusplus

#if __has_c_attribute(omp::sequence) != SUPPORTED
#error "No idea what you're talking about"
#endif

#if __has_c_attribute(omp::directive) != SUPPORTED
#error "No idea what you're talking about"
#endif

#if __has_c_attribute(omp::totally_bogus)
#error "No idea what you're talking about"
#endif

#endif

