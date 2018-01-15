.. raw:: html

  <style type="text/css">
    .none { background-color: #FFCCCC }
    .partial { background-color: #FFFF99 }
    .good { background-color: #CCFF99 }
  </style>

.. role:: none
.. role:: partial
.. role:: good

==================
OpenMP Support
==================

Clang fully supports OpenMP 3.1 + some elements of OpenMP 4.5. Clang supports offloading to X86_64, AArch64 and PPC64[LE] devices.
Support for Cuda devices is not ready yet.
The status of major OpenMP 4.5 features support in Clang.

Standalone directives
=====================

* #pragma omp [for] simd: :good:`Complete`.

* #pragma omp declare simd: :partial:`Partial`.  We support parsing/semantic
  analysis + generation of special attributes for X86 target, but still
  missing the LLVM pass for vectorization.

* #pragma omp taskloop [simd]: :good:`Complete`.

* #pragma omp target [enter|exit] data: :good:`Complete`.

* #pragma omp target update: :good:`Complete`.

* #pragma omp target: :good:`Complete`.

* #pragma omp declare target: :partial:`Partial`.  No full codegen support.

* #pragma omp teams: :good:`Complete`.

* #pragma omp distribute [simd]: :good:`Complete`.

* #pragma omp distribute parallel for [simd]: :good:`Complete`.

Combined directives
===================

* #pragma omp parallel for simd: :good:`Complete`.

* #pragma omp target parallel: :partial:`Partial`.  No support for the `depend` clauses.

* #pragma omp target parallel for [simd]: :partial:`Partial`.  No support for the `depend` clauses.

* #pragma omp target simd: :partial:`Partial`.  No support for the `depend` clauses.

* #pragma omp target teams: :partial:`Partial`.  No support for the `depend` clauses.

* #pragma omp teams distribute [simd]: :good:`Complete`.

* #pragma omp target teams distribute [simd]: :partial:`Partial`.  No support for the and `depend` clauses.

* #pragma omp teams distribute parallel for [simd]: :good:`Complete`.

* #pragma omp target teams distribute parallel for [simd]: :partial:`Partial`.  No full codegen support.

Clang does not support any constructs/updates from upcoming OpenMP 5.0 except for `reduction`-based clauses in the `task` and `target`-based directives.

