.. raw:: html

  <style type="text/css">
    .none { background-color: #FFCCCC }
    .part { background-color: #FFFF99 }
    .good { background-color: #CCFF99 }
  </style>

.. role:: none
.. role:: part
.. role:: good

.. contents::
   :local:

OpenMP Support
==============

Clang fully supports OpenMP 4.5. Clang supports offloading to X86_64, AArch64,
PPC64[LE] and has `basic support for Cuda devices`_.

* #pragma omp declare simd: :part:`Partial`.  We support parsing/semantic
  analysis + generation of special attributes for X86 target, but still
  missing the LLVM pass for vectorization.

In addition, the LLVM OpenMP runtime `libomp` supports the OpenMP Tools
Interface (OMPT) on x86, x86_64, AArch64, and PPC64 on Linux, Windows, and macOS.

For the list of supported features from OpenMP 5.0 see `OpenMP implementation details`_.

General improvements
====================
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

- When using the collapse clause on a loop nest the default behavior
  is to automatically extend the representation of the loop counter to
  64 bits for the cases where the sizes of the collapsed loops are not
  known at compile time. To prevent this conservative choice and use
  at most 32 bits, compile your program with the
  `-fopenmp-optimistic-collapse`.

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


.. _OpenMP implementation details:

OpenMP 5.0 Implementation Details
=================================

The following table provides a quick overview over various OpenMP 5.0 features
and their implementation status. Please contact *openmp-dev* at
*lists.llvm.org* for more information or if you want to help with the
implementation.

+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
|Category                      | Feature                                                      | Status                   | Reviews                                                               |
+==============================+==============================================================+==========================+=======================================================================+
| loop extension               | support != in the canonical loop form                        | :good:`done`             | D54441                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| loop extension               | #pragma omp loop (directive)                                 | :part:`worked on`        |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| loop extension               | collapse imperfectly nested loop                             | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| loop extension               | collapse non-rectangular nested loop                         | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| loop extension               | C++ range-base for loop                                      | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| loop extension               | clause: if for SIMD directives                               | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| loop extension               | inclusive scan extension (matching C++17 PSTL)               | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| memory mangagement           | memory allocators                                            | :good:`done`             | r341687,r357929                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| memory mangagement           | allocate directive and allocate clause                       | :good:`done`             | r355614,r335952                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| OMPD                         | OMPD interfaces                                              | :part:`not upstream`     | https://github.com/OpenMPToolsInterface/LLVM-openmp/tree/ompd-tests   |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| OMPT                         | OMPT interfaces                                              | :part:`mostly done`      |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| thread affinity extension    | thread affinity extension                                    | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| task extension               | taskloop reduction                                           | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| task extension               | task affinity                                                | :part:`not upstream`     |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| task extension               | clause: depend on the taskwait construct                     | :part:`worked on`        |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| task extension               | depend objects and detachable tasks                          | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| task extension               | mutexinoutset dependence-type for tasks                      | :good:`done`             | D53380,D57576                                                         |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| task extension               | combined taskloop constructs                                 | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| task extension               | master taskloop                                              | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| task extension               | parallel master taskloop                                     | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| task extension               | master taskloop simd                                         | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| task extension               | parallel master taskloop simd                                | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| SIMD extension               | atomic and simd constructs inside SIMD code                  | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| SIMD extension               | SIMD nontemporal                                             | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | infer target functions from initializers                     | :part:`worked on`        |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | infer target variables from initializers                     | :part:`worked on`        |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | OMP_TARGET_OFFLOAD environment variable                      | :good:`done`             | D50522                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | support full 'defaultmap' functionality                      | :good:`done`             | D69204                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | device specific functions                                    | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | clause: device_type                                          | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | clause: extended device                                      | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | clause: uses_allocators clause                               | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | clause: in_reduction                                         | :part:`worked on`        | r308768                                                               |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | omp_get_device_num()                                         | :part:`worked on`        | D54342                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | structure mapping of references                              | :none:`unclaimed`        |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | nested target declare                                        | :good:`done`             | D51378                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | implicitly map 'this' (this[:1])                             | :good:`done`             | D55982                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | allow access to the reference count (omp_target_is_present)  | :part:`worked on`        |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | requires directive                                           | :part:`partial`          |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | clause: unified_shared_memory                                | :good:`done`             | D52625,D52359                                                         |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | clause: unified_address                                      | :part:`partial`          |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | clause: reverse_offload                                      | :none:`unclaimed parts`  | D52780                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | clause: atomic_default_mem_order                             | :good:`done`             | D53513                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | clause: dynamic_allocators                                   | :part:`unclaimed parts`  | D53079                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | user-defined mappers                                         | :part:`worked on`        | D56326,D58638,D58523,D58074,D60972,D59474                             |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | mapping lambda expression                                    | :good:`done`             | D51107                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | clause: use_device_addr for target data                      | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | support close modifier on map clause                         | :good:`done`             | D55719,D55892                                                         |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | teams construct on the host device                           | :part:`worked on`        | Clang part is done, r371553.                                          |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | support non-contiguous array sections for target update      | :part:`worked on`        |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| device extension             | pointer attachment                                           | :none:`unclaimed`        |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| atomic extension             | hints for the atomic construct                               | :good:`done`             | D51233                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| base language                | C11 support                                                  | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| base language                | C++11/14/17 support                                          | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| base language                | lambda support                                               | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| misc extension               | array shaping                                                | :good:`done`             | D74144                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| misc extension               | library shutdown (omp_pause_resource[_all])                  | :none:`unclaimed parts`  | D55078                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| misc extension               | metadirectives                                               | :part:`worked on`        |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| misc extension               | conditional modifier for lastprivate clause                  | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| misc extension               | iterator and multidependences                                | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| misc extension               | depobj directive and depobj dependency kind                  | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| misc extension               | user-defined function variants                               | :part:`worked on`        | D67294, D64095, D71847, D71830                                        |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| misc extension               | pointer/reference to pointer based array reductions          | :none:`unclaimed`        |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| misc extension               | prevent new type definitions in clauses                      | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| memory model extension       | memory model update (seq_cst, acq_rel, release, acquire,...) | :good:`done`             |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+


OpenMP 5.1 Implementation Details
=================================

The following table provides a quick overview over various OpenMP 5.1 features
and their implementation status, as defined in the technical report 8 (TR8).
Please contact *openmp-dev* at *lists.llvm.org* for more information or if you
want to help with the implementation.

+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
|Category                      | Feature                                                      | Status                   | Reviews                                                               |
+==============================+==============================================================+==========================+=======================================================================+
| misc extension               | user-defined function variants with #ifdef protection        | :part:`worked on`        | D71179                                                                |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| misc extension               | default(firstprivate) & default(private)                     | :part:`worked on`        |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
| loop extension               | Loop tiling transformation                                   | :part:`claimed`          |                                                                       |
+------------------------------+--------------------------------------------------------------+--------------------------+-----------------------------------------------------------------------+
