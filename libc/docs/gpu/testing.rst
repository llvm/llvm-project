.. _libc_gpu_testing:


============================
Testing the GPU libc library
============================

.. note::
   Running GPU tests with high parallelism is likely to cause spurious failures,
   out of resource errors, or indefinite hangs. limiting the number of threads
   used while testing using ``LIBC_GPU_TEST_JOBS=<N>`` is highly recommended.

.. contents:: Table of Contents
  :depth: 4
  :local:

Testing Infrastructure
======================

The testing support in LLVM's libc implementation for GPUs is designed to mimic
the standard unit tests as much as possible. We use the :ref:`libc_gpu_rpc`
support to provide the necessary utilities like printing from the GPU. Execution
is performed by emitting a ``_start`` kernel from the GPU
that is then called by an external loader utility. This is an example of how
this can be done manually:

.. code-block:: sh

   $> clang++ crt1.o test.cpp --target=amdgcn-amd-amdhsa -mcpu=gfx90a -flto
   $> ./amdhsa_loader --threads 1 --blocks 1 a.out
   Test Passed!

Unlike the exported ``libcgpu.a``, the testing architecture can only support a
single architecture at a time. This is either detected automatically, or set
manually by the user using ``LIBC_GPU_TEST_ARCHITECTURE``. The latter is useful
in cases where the user does not build LLVM's libc on machine with the GPU to
use for testing.
