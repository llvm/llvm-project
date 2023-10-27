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
using the ``LLVM_ENABLE_RUNTIMES=libc`` option. To enable libc for the GPU,
enable the ``LIBC_GPU_BUILD`` option. By default, ``libcgpu.a`` will be built
using every supported GPU architecture. To restrict the number of architectures
build, either set ``LIBC_GPU_ARCHITECTURES`` to the list of desired
architectures manually or use ``native`` to detect the GPUs on your system. A
typical ``cmake`` configuration will look like this:

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build
  $> cd build
  $> cmake ../llvm -G Ninja                                \
     -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt"        \
     -DLLVM_ENABLE_RUNTIMES="libc;openmp"                  \
     -DCMAKE_BUILD_TYPE=<Debug|Release>   \ # Select build type
     -DLIBC_GPU_BUILD=ON                  \ # Build in GPU mode
     -DLIBC_GPU_ARCHITECTURES=all         \ # Build all supported architectures
     -DCMAKE_INSTALL_PREFIX=<PATH>        \ # Where 'libcgpu.a' will live
  $> ninja install

Since we want to include ``clang``, ``lld`` and ``compiler-rt`` in our
toolchain, we list them in ``LLVM_ENABLE_PROJECTS``. To ensure ``libc`` is built
using a compatible compiler and to support ``openmp`` offloading, we list them
in ``LLVM_ENABLE_RUNTIMES`` to build them after the enabled projects using the
newly built compiler. ``CMAKE_INSTALL_PREFIX`` specifies the installation
directory in which to install the ``libcgpu.a`` library and headers along with
LLVM. The generated headers will be placed in ``include/gpu-none-llvm``.

Usage
=====

Once the ``libcgpu.a`` static archive has been built it can be linked directly
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
  arch            gfx90a
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
