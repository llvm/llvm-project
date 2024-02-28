.. _libc_gpu_usage:


===================
Using libc for GPUs
===================

.. contents:: Table of Contents
  :depth: 4
  :local:

Building the GPU library
========================

LLVM's libc GPU support *must* be built with an up-to-date ``clang`` compiler
due to heavy reliance on ``clang``'s GPU support. This can be done automatically
using the LLVM runtimes support. The GPU build is done using cross-compilation
to the GPU architecture. This project currently supports AMD and NVIDIA GPUs
which can be targeted using the appropriate target name. The following
invocation will enable a cross-compiling build for the GPU architecture and
enable the ``libc`` project only for them.

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build
  $> cd build
  $> cmake ../llvm -G Ninja                                               \
     -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt"                       \
     -DLLVM_ENABLE_RUNTIMES="openmp"                                      \
     -DCMAKE_BUILD_TYPE=<Debug|Release>   \ # Select build type
     -DCMAKE_INSTALL_PREFIX=<PATH>        \ # Where 'libcgpu.a' will live
     -DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES=libc             \
     -DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES=libc               \
     -DLLVM_RUNTIME_TARGETS=default;amdgcn-amd-amdhsa;nvptx64-nvidia-cuda
  $> ninja install

Since we want to include ``clang``, ``lld`` and ``compiler-rt`` in our
toolchain, we list them in ``LLVM_ENABLE_PROJECTS``. To ensure ``libc`` is built
using a compatible compiler and to support ``openmp`` offloading, we list them
in ``LLVM_ENABLE_RUNTIMES`` to build them after the enabled projects using the
newly built compiler. ``CMAKE_INSTALL_PREFIX`` specifies the installation
directory in which to install the ``libcgpu-nvptx.a`` and ``libcgpu-amdgpu.a``
libraries and headers along with LLVM. The generated headers will be placed in
``include/<gpu-triple>``.

Usage
=====

Once the static archive has been built it can be linked directly
with offloading applications as a standard library. This process is described in
the `clang documentation <https://clang.llvm.org/docs/OffloadingDesign.html>`_.
This linking mode is used by the OpenMP toolchain, but is currently opt-in for
the CUDA and HIP toolchains through the ``--offload-new-driver``` and
``-fgpu-rdc`` flags. A typical usage will look this this:

.. code-block:: sh

  $> clang foo.c -fopenmp --offload-arch=gfx90a -lcgpu

The ``libcgpu.a`` static archive is a fat-binary containing LLVM-IR for each
supported target device. The supported architectures can be seen using LLVM's
``llvm-objdump`` with the ``--offloading`` flag:

.. code-block:: sh

  $> llvm-objdump --offloading libcgpu.a
  libcgpu.a(strcmp.cpp.o):    file format elf64-x86-64

  OFFLOADING IMAGE [0]:
  kind            llvm ir
  arch            generic
  triple          amdgcn-amd-amdhsa
  producer        none

Because the device code is stored inside a fat binary, it can be difficult to
inspect the resulting code. This can be done using the following utilities:

.. code-block:: sh

   $> llvm-ar x libcgpu.a strcmp.cpp.o
   $> clang-offload-packager strcmp.cpp.o --image=arch=gfx90a,file=gfx90a.bc
   $> opt -S out.bc
   ...

Please note that this fat binary format is provided for compatibility with
existing offloading toolchains. The implementation in ``libc`` does not depend
on any existing offloading languages and is completely freestanding.
