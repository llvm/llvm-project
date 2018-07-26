.. raw:: html

  <style type="text/css">
    .none { background-color: #FFCCCC }
    .partial { background-color: #FFFF99 }
    .good { background-color: #CCFF99 }
  </style>

.. role:: none
.. role:: partial
.. role:: good

.. contents::
   :local:

==================
OpenMP Support
==================

Clang fully supports OpenMP 4.5. Clang supports offloading to X86_64, AArch64,
PPC64[LE] and has `basic support for Cuda devices`_.

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

* #pragma omp declare target: :good:`Complete`.

* #pragma omp teams: :good:`Complete`.

* #pragma omp distribute [simd]: :good:`Complete`.

* #pragma omp distribute parallel for [simd]: :good:`Complete`.

Combined directives
===================

* #pragma omp parallel for simd: :good:`Complete`.

* #pragma omp target parallel: :good:`Complete`.

* #pragma omp target parallel for [simd]: :good:`Complete`.

* #pragma omp target simd: :good:`Complete`.

* #pragma omp target teams: :good:`Complete`.

* #pragma omp teams distribute [simd]: :good:`Complete`.

* #pragma omp target teams distribute [simd]: :good:`Complete`.

* #pragma omp teams distribute parallel for [simd]: :good:`Complete`.

* #pragma omp target teams distribute parallel for [simd]: :good:`Complete`.

Clang does not support any constructs/updates from upcoming OpenMP 5.0 except
for `reduction`-based clauses in the `task` and `target`-based directives.

In addition, the LLVM OpenMP runtime `libomp` supports the OpenMP Tools
Interface (OMPT) on x86, x86_64, AArch64, and PPC64 on Linux, Windows, and mac OS.
ows, and mac OS.

.. _basic support for Cuda devices:

Cuda devices support
====================

Directives execution modes
--------------------------

Clang code generation for target regions supports two modes: the SPMD and
non-SPMD modes. Clang chooses one of these two modes automatically based on the
way directives and clauses on those directives are used. The SPMD mode uses a
simplified set of runtime functions thus increasing performance at the cost of
supporting some OpenMP features. The non-SPMD mode is the most generic mode and
supports all currently available OpenMP features. The compiler will always
attempt to use the SPMD mode wherever possible. SPMD mode will not be used if:

   - The target region contains an `if()` clause that refers to a `parallel`
     directive.

   - The target region contains a `parallel` directive with a `num_threads()`
     clause.

   - The target region contains user code (other than OpenMP-specific
     directives) in between the `target` and the `parallel` directives.

Data-sharing modes
------------------

Clang supports two data-sharing models for Cuda devices: `Generic` and `Cuda`
modes. The default mode is `Generic`. `Cuda` mode can give an additional
performance and can be activated using the `-fopenmp-cuda-mode` flag. In
`Generic` mode all local variables that can be shared in the parallel regions
are stored in the global memory. In `Cuda` mode local variables are not shared
between the threads and it is user responsibility to share the required data
between the threads in the parallel regions.

Features not supported or with limited support for Cuda devices
---------------------------------------------------------------

- Reductions across the teams are not supported yet.

- Cancellation constructs are not supported.

- Doacross loop nest is not supported.

- User-defined reductions are supported only for trivial types.

- Nested parallelism: inner parallel regions are executed sequentially.

- Static linking of libraries containing device code is not supported yet.

- Automatic translation of math functions in target regions to device-specific
  math functions is not implemented yet.

- Debug information for OpenMP target regions is not supported yet.

