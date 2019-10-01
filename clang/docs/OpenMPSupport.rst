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

Clang supports the following OpenMP 5.0 features

* The `reduction`-based clauses in the `task` and `target`-based directives.

* Support relational-op != (not-equal) as one of the canonical forms of random
  access iterator.

* Support for mapping of the lambdas in target regions.

* Parsing/sema analysis for the requires directive.

* Nested declare target directives.

* Make the `this` pointer implicitly mapped as `map(this[:1])`.

* The `close` *map-type-modifier*.

Clang fully supports OpenMP 4.5. Clang supports offloading to X86_64, AArch64,
PPC64[LE] and has `basic support for Cuda devices`_.

* #pragma omp declare simd: :partial:`Partial`.  We support parsing/semantic
  analysis + generation of special attributes for X86 target, but still
  missing the LLVM pass for vectorization.

In addition, the LLVM OpenMP runtime `libomp` supports the OpenMP Tools
Interface (OMPT) on x86, x86_64, AArch64, and PPC64 on Linux, Windows, and macOS.

General improvements
--------------------
- New collapse clause scheme to avoid expensive remainder operations.
  Compute loop index variables after collapsing a loop nest via the
  collapse clause by replacing the expensive remainder operation with
  multiplications and additions.

- The default schedules for the `distribute` and `for` constructs in a
  parallel region and in SPMD mode have changed to ensure coalesced
  accesses. For the `distribute` construct, a static schedule is used
  with a chunk size equal to the number of threads per team (default
  value of threads or as specified by the `thread_limit` clause if
  present). For the `for` construct, the schedule is static with chunk
  size of one.
  
- Simplified SPMD code generation for `distribute parallel for` when
  the new default schedules are applicable.

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

Collapsed loop nest counter
---------------------------

When using the collapse clause on a loop nest the default behavior is to
automatically extend the representation of the loop counter to 64 bits for
the cases where the sizes of the collapsed loops are not known at compile
time. To prevent this conservative choice and use at most 32 bits,
compile your program with the `-fopenmp-optimistic-collapse`.


Features not supported or with limited support for Cuda devices
---------------------------------------------------------------

- Cancellation constructs are not supported.

- Doacross loop nest is not supported.

- User-defined reductions are supported only for trivial types.

- Nested parallelism: inner parallel regions are executed sequentially.

- Static linking of libraries containing device code is not supported yet.

- Automatic translation of math functions in target regions to device-specific
  math functions is not implemented yet.

- Debug information for OpenMP target regions is supported, but sometimes it may
  be required to manually specify the address class of the inspected variables.
  In some cases the local variables are actually allocated in the global memory,
  but the debug info may be not aware of it.

