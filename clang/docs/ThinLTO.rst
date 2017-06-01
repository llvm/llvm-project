=======
ThinLTO
=======

.. contents::
   :local:

Introduction
============

*ThinLTO* compilation is a new type of LTO that is both scalable and
incremental. *LTO* (Link Time Optimization) achieves better
runtime performance through whole-program analysis and cross-module
optimization. However, monolithic LTO implements this by merging all
input into a single module, which is not scalable
in time or memory, and also prevents fast incremental compiles.

In ThinLTO mode, as with regular LTO, clang emits LLVM bitcode after the
compile phase. The ThinLTO bitcode is augmented with a compact summary
of the module. During the link step, only the summaries are read and
merged into a combined summary index, which includes an index of function
locations for later cross-module function importing. Fast and efficient
whole-program analysis is then performed on the combined summary index.

However, all transformations, including function importing, occur
later when the modules are optimized in fully parallel backends.
By default, linkers_ that support ThinLTO are set up to launch
the ThinLTO backends in threads. So the usage model is not affected
as the distinction between the fast serial thin link step and the backends
is transparent to the user.

For more information on the ThinLTO design and current performance,
see the LLVM blog post `ThinLTO: Scalable and Incremental LTO
<http://blog.llvm.org/2016/06/thinlto-scalable-and-incremental-lto.html>`_.
While tuning is still in progress, results in the blog post show that
ThinLTO already performs well compared to LTO, in many cases matching
the performance improvement.

Current Status
==============

Clang/LLVM
----------
.. _compiler:

The 3.9 release of clang includes ThinLTO support. However, ThinLTO
is under active development, and new features, improvements and bugfixes
are being added for the next release. For the latest ThinLTO support,
`build a recent version of clang and LLVM
<http://llvm.org/docs/CMake.html>`_.

Linkers
-------
.. _linkers:
.. _linker:

ThinLTO is currently supported for the following linkers:

- **gold (via the gold-plugin)**:
  Similar to monolithic LTO, this requires using
  a `gold linker configured with plugins enabled
  <http://llvm.org/docs/GoldPlugin.html>`_.
- **ld64**:
  Starting with `Xcode 8 <https://developer.apple.com/xcode/>`_.
- **lld**:
  Starting with r284050 (ELF only).

Usage
=====

Basic
-----

To utilize ThinLTO, simply add the -flto=thin option to compile and link. E.g.

.. code-block:: console

  % clang -flto=thin -O2 file1.c file2.c -c
  % clang -flto=thin -O2 file1.o file2.o -o a.out

As mentioned earlier, by default the linkers will launch the ThinLTO backend
threads in parallel, passing the resulting native object files back to the
linker for the final native link.  As such, the usage model the same as
non-LTO.

With gold, if you see an error during the link of the form:

.. code-block:: console

  /usr/bin/ld: error: /path/to/clang/bin/../lib/LLVMgold.so: could not load plugin library: /path/to/clang/bin/../lib/LLVMgold.so: cannot open shared object file: No such file or directory

Then either gold was not configured with plugins enabled, or clang
was not built with ``-DLLVM_BINUTILS_INCDIR`` set properly. See
the instructions for the
`LLVM gold plugin <http://llvm.org/docs/GoldPlugin.html#how-to-build-it>`_.

Controlling Backend Parallelism
-------------------------------
.. _parallelism:

By default, the ThinLTO link step will launch up to
``std::thread::hardware_concurrency`` number of threads in parallel.
For machines with hyper-threading, this is the total number of
virtual cores. For some applications and machine configurations this
may be too aggressive, in which case the amount of parallelism can
be reduced to ``N`` via:

- gold:
  ``-Wl,-plugin-opt,jobs=N``
- ld64:
  ``-Wl,-mllvm,-threads=N``
- lld:
  ``-Wl,--thinlto-jobs=N``

Incremental
-----------
.. _incremental:

ThinLTO supports fast incremental builds through the use of a cache,
which currently must be enabled through a linker option.

- gold (as of LLVM r279883):
  ``-Wl,-plugin-opt,cache-dir=/path/to/cache``
- ld64 (support in clang 3.9 and Xcode 8):
  ``-Wl,-cache_path_lto,/path/to/cache``
- lld (as of LLVM r296702):
  ``-Wl,--thinlto-cache-dir=/path/to/cache``

Clang Bootstrap
---------------

To bootstrap clang/LLVM with ThinLTO, follow these steps:

1. The host compiler_ must be a version of clang that supports ThinLTO.
#. The host linker_ must support ThinLTO (and in the case of gold, must be
   `configured with plugins enabled <http://llvm.org/docs/GoldPlugin.html>`_.
#. Use the following additional `CMake variables
   <http://llvm.org/docs/CMake.html#options-and-variables>`_
   when configuring the bootstrap compiler build:

  * ``-DLLVM_ENABLE_LTO=Thin``
  * ``-DLLVM_PARALLEL_LINK_JOBS=1``
    (since the ThinLTO link invokes parallel backend jobs)
  * ``-DCMAKE_C_COMPILER=/path/to/host/clang``
  * ``-DCMAKE_CXX_COMPILER=/path/to/host/clang++``
  * ``-DCMAKE_RANLIB=/path/to/host/llvm-ranlib``
  * ``-DCMAKE_AR=/path/to/host/llvm-ar``

#. To use additional linker arguments for controlling the backend
   parallelism_ or enabling incremental_ builds of the bootstrap compiler,
   after configuring the build, modify the resulting CMakeCache.txt file in the
   build directory. Specify any additional linker options after
   ``CMAKE_EXE_LINKER_FLAGS:STRING=``. Note the configure may fail if
   linker plugin options are instead specified directly in the previous step.

More Information
================

* From LLVM project blog:
  `ThinLTO: Scalable and Incremental LTO
  <http://blog.llvm.org/2016/06/thinlto-scalable-and-incremental-lto.html>`_
