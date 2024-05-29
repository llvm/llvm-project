.. _libc_gpu_testing:


=========================
Testing the GPU C library
=========================

.. note::
   Running GPU tests with high parallelism is likely to cause spurious failures,
   out of resource errors, or indefinite hangs. limiting the number of threads
   used while testing using ``LIBC_GPU_TEST_JOBS=<N>`` is highly recommended.

.. contents:: Table of Contents
  :depth: 4
  :local:

Testing infrastructure
======================

The LLVM C library supports different kinds of :ref:`tests <build_and_test>`
depending on the build configuration. The GPU target is considered a full build
and therefore provides all of its own utilities to build and run the generated
tests. Currently the GPU supports two kinds of tests.

#. **Hermetic tests** - These are unit tests built with a test suite similar to
   Google's ``gtest`` infrastructure. These use the same infrastructure as unit
   tests except that the entire environment is self-hosted. This allows us to
   run them on the GPU using our custom utilities. These are used to test the
   majority of functional implementations.

#. **Integration tests** - These are lightweight tests that simply call a
   ``main`` function and checks if it returns non-zero. These are primarily used
   to test interfaces that are sensitive to threading.

The GPU uses the same testing infrastructure as the other supported ``libc``
targets. We do this by treating the GPU as a standard hosted environment capable
of launching a ``main`` function. Effectively, this means building our own
startup libraries and loader.

Testing utilities
=================

We provide two utilities to execute arbitrary programs on the GPU. That is the
``loader`` and the ``start`` object.

Startup object
--------------

This object mimics the standard object used by existing C library
implementations. Its job is to perform the necessary setup prior to calling the
``main`` function. In the GPU case, this means exporting GPU kernels that will
perform the necessary operations. Here we use ``_begin`` and ``_end`` to handle
calling global constructors and destructors while ``_start`` begins the standard
execution. The following code block shows the implementation for AMDGPU
architectures.

.. code-block:: c++

  extern "C" [[gnu::visibility("protected"), clang::amdgpu_kernel]] void
  _begin(int argc, char **argv, char **env) {
    LIBC_NAMESPACE::atexit(&LIBC_NAMESPACE::call_fini_array_callbacks);
    LIBC_NAMESPACE::call_init_array_callbacks(argc, argv, env);
  }

  extern "C" [[gnu::visibility("protected"), clang::amdgpu_kernel]] void
  _start(int argc, char **argv, char **envp, int *ret) {
    __atomic_fetch_or(ret, main(argc, argv, envp), __ATOMIC_RELAXED);
  }

  extern "C" [[gnu::visibility("protected"), clang::amdgpu_kernel]] void
  _end(int retval) {
    LIBC_NAMESPACE::exit(retval);
  }

Loader runtime
--------------

The startup object provides a GPU executable with callable kernels for the
respective runtime. We can then define a minimal runtime that will launch these
kernels on the given device. Currently we provide the ``amdhsa-loader`` and
``nvptx-loader`` targeting the AMD HSA runtime and CUDA driver runtime
respectively. By default these will launch with a single thread on the GPU.

.. code-block:: sh

   $> clang++ crt1.o test.cpp --target=amdgcn-amd-amdhsa -mcpu=native -flto
   $> amdhsa_loader --threads 1 --blocks 1 ./a.out
   Test Passed!

The loader utility will forward any arguments passed after the executable image
to the program on the GPU as well as any set environment variables. The number
of threads and blocks to be set can be controlled with ``--threads`` and
``--blocks``. These also accept additional ``x``, ``y``, ``z`` variants for
multidimensional grids.

Running tests
=============

Tests will only be built and run if a GPU target architecture is set and the
corresponding loader utility was built. These can be overridden with the
``LIBC_GPU_TEST_ARCHITECTURE`` and ``LIBC_GPU_LOADER_EXECUTABLE`` :ref:`CMake
options <gpu_cmake_options>`. Once built, they can be run like any other tests.
The CMake target depends on how the library was built.

#. **Cross build** - If the C library was built using ``LLVM_ENABLE_PROJECTS``
   or a runtimes cross build, then the standard targets will be present in the
   base CMake build directory.

   #. All tests - You can run all supported tests with the command:

      .. code-block:: sh

        $> ninja check-libc

   #. Hermetic tests - You can run hermetic with tests the command:

      .. code-block:: sh

        $> ninja libc-hermetic-tests

   #. Integration tests - You can run integration tests by the command:

      .. code-block:: sh

        $> ninja libc-integration-tests

#. **Runtimes build** - If the library was built using ``LLVM_ENABLE_RUNTIMES``
   then the actual ``libc`` build will be in a separate directory.

   #. All tests - You can run all supported tests with the command:

      .. code-block:: sh

        $> ninja check-libc-amdgcn-amd-amdhsa
        $> ninja check-libc-nvptx64-nvidia-cuda

   #. Specific tests - You can use the same targets as above by entering the
      runtimes build directory.

      .. code-block:: sh

        $> ninja -C runtimes/runtimes-amdgcn-amd-amdhsa-bins check-libc
        $> ninja -C runtimes/runtimes-nvptx64-nvidia-cuda-bins check-libc
        $> cd runtimes/runtimes-amdgcn-amd-amdhsa-bins && ninja check-libc
        $> cd runtimes/runtimes-nvptx64-nvidia-cuda-bins && ninja check-libc

Tests can also be built and run manually using the respective loader utility.
