.. _libc_gpu_motivation:

==========================
Motivation and Limitations
==========================

.. contents:: Table of Contents
  :depth: 4
  :local:

Motivation
==========

This project aims to provide a large subset of the C standard library to users
of GPU accelerators. We deliberately choose to only implement a subset of the C
library as some features are not expressly useful or easily implemented on the
GPU. This will be discussed further in `Limitations <libc_gpu_limitations>`_.
The main motivation behind this project is to provide the well understood C
library as a firm base for GPU development.

The main idea behind this project is that programming GPUs can be as
straightforward as programming on CPUs. This project aims to validate the GPU as
a more general-purpose target. The implementations here will also enable more
complex implementations of other libraries on the GPU, such as ``libc++``.

Host services and C library features are currently provided sparsely by the
different GPU vendors. We wish to provide these functions more completely and
make their implementations available in a common format. This is useful for
targets like OpenMP offloading or SYCL which wish to unify the offloading
toolchain. We also aim to provide these functions in a format compatible with
offloading in ``Clang`` so that we can treat the C library for the GPU as a
standard static library.

A valuable use for providing C library features on the GPU is for testing. For
this reason we build `tests on the GPU <libc_gpu_testing>`_ that can run a unit
test as if it were being run on the CPU. This also helps users port applications
that traditionally were run on the CPU. With this support, we can expand test
coverage for the GPU backend to the existing LLVM C library tests.

.. _libc_gpu_limitations:

Limitations
===========

We only implement a subset of the standard C library. The GPU does not
currently support thread local variables in all cases, so variables like
``errno`` are not provided. Furthermore, the GPU under the OpenCL execution
model cannot safely provide a mutex interface. This means that features like
file buffering are not implemented on the GPU. We can also not easily provide
threading features on the GPU due to the execution model so these will be
ignored, as will features like ``locale`` or ``time``.
