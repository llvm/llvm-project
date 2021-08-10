// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fsyntax-only -verify=pre -Wpre-openmp-51-compat %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fsyntax-only -verify=off %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fsyntax-only -verify=ext -Wopenmp %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fsyntax-only -verify=ext -Wopenmp-51-extensions %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fsyntax-only -verify=ext -Wall %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fsyntax-only -verify=ext %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fsyntax-only -verify=off -Wno-openmp %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fsyntax-only -verify=off -Wno-openmp-51-extensions %s

// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fsyntax-only -verify=pre -Wpre-openmp-51-compat -x c -fdouble-square-bracket-attributes %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fsyntax-only -verify=off -x c -std=c2x %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fsyntax-only -verify=ext -Wopenmp -x c -std=c2x %s

// off-no-diagnostics

int x;
[[omp::directive(threadprivate(x))]]; // pre-warning {{specifying OpenMP directives with [[]] is incompatible with OpenMP standards before OpenMP 5.1}} \
                                      // ext-warning {{specifying OpenMP directives with [[]] is an OpenMP 5.1 extension}}

