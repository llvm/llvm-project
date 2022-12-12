.. _GPU_mode:

==============
GPU Mode
==============

.. include:: check.rst

.. contents:: Table of Contents
  :depth: 4
  :local:

.. note:: This feature is very experimental and may change in the future.

The *GPU* mode of LLVM's libc is an experimental mode used to support calling
libc routines during GPU execution. The goal of this project is to provide
access to the standard C library on systems running accelerators. To begin using
this library, build and install the ``libcgpu.a`` static archive following the
instructions in :ref:`building_gpu_mode` and link with your offloading
application.

.. _building_gpu_mode:

Building the GPU library
========================

LLVM's libc GPU support *must* be built using the same compiler as the final
application to ensure relative LLVM bitcode compatibility. This can be done
automatically using the ``LLVM_ENABLE_RUNTIMES=libc`` option. Furthermore,
building for the GPU is only supported in :ref:`fullbuild_mode`. To enable the
GPU build, set the target OS to ``gpu`` via ``LLVM_LIBC_TARGET_OS=gpu``. By
default, ``libcgpu.a`` will be built using every supported GPU architecture. To
restrict the number of architectures build, set ``LLVM_LIBC_GPU_ARCHITECTURES``
to the list of desired architectures or use ``all``. A typical ``cmake``
configuration will look like this:

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build
  $> cd build
  $> cmake ../llvm -G Ninja                                \
     -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt"        \
     -DLLVM_ENABLE_RUNTIMES="libc;openmp"                  \
     -DCMAKE_BUILD_TYPE=<Debug|Release>  \ # Select build type
     -DLLVM_LIBC_FULL_BUILD=ON           \ # We need the full libc
     -DLLVM_LIBC_TARGET_OS=gpu           \ # Build in GPU mode
     -DLLVM_LIBC_GPU_ARCHITECTURES=all   \ # Build all supported architectures
     -DCMAKE_INSTALL_PREFIX=<PATH>       \ # Where 'libcgpu.a' will live
  $> ninja install

Since we want to include ``clang``, ``lld`` and ``compiler-rt`` in our
toolchain, we list them in ``LLVM_ENABLE_PROJECTS``. To ensure ``libc`` is built
using a compatible compiler and to support ``openmp`` offloading, we list them
in ``LLVM_ENABLE_RUNTIMES`` to build them after the enabled projects using the
newly built compiler. ``CMAKE_INSTALL_PREFIX`` specifies the installation
directory in which to install the ``libcgpu.a`` library along with LLVM.

Usage
=====

Once the ``libcgpu.a`` static archive has been built in
:ref:`building_gpu_mode`, it can be linked directly with offloading applications
as a standard library. This process is described in the `clang documentation
<https://clang.llvm.org/docs/OffloadingDesign.html>_`. This linking mode is used
by the OpenMP toolchain, but is currently opt-in for the CUDA and HIP toolchains
using the ``--offload-new-driver``` and ``-fgpu-rdc`` flags. A typical usage
will look this this:

.. code-block:: sh

  $> clang foo.c -fopenmp --offload-arch=gfx90a -lcgpu

The ``libcgpu.a`` static archive is a fat-binary containing LLVM-IR for each
supported target device. The supported architectures can be seen using LLVM's
objdump with the ``--offloading`` flag:

.. code-block:: sh

  $> llvm-objdump --offloading libcgpu.a
  libcgpu.a(strcmp.cpp.o):    file format elf64-x86-64

  OFFLOADING IMAGE [0]:
  kind            llvm ir
  arch            gfx90a
  triple          amdgcn-amd-amdhsa
  producer        <none>

Because the device code is stored inside a fat binary, it can be difficult to
inspect the resulting code. This can be done using the following utilities:

.. code-block:: sh

   $> llvm-ar x libcgpu.a strcmp.cpp.o
   $> clang-offload-packager strcmp.cpp.o --image=arch=gfx90a,file=gfx90a.bc
   $> opt -S out.bc
   ...

Supported Functions
===================

The following functions and headers are supported at least partially on the
device. Currently, only basic device functions that do not require an operating
system are supported on the device. Supporting functions like `malloc` using an
RPC mechanism is a work-in-progress.

ctype.h
-------

=============  =========
Function Name  Available
=============  =========
isalnum        |check|
isalpha        |check|
isascii        |check|
isblank        |check|
iscntrl        |check|
isdigit        |check|
isgraph        |check|
islower        |check|
isprint        |check|
ispunct        |check|
isspace        |check|
isupper        |check|
isxdigit       |check|
toascii        |check|
tolower        |check|
toupper        |check|
=============  =========

string.h
--------

=============   =========
Function Name   Available
=============   =========
bcmp            |check|
bzero           |check|
memccpy         |check|
memchr          |check|
memcmp          |check|
memcpy          |check|
memmove         |check|
mempcpy         |check|
memrchr         |check|
memset          |check|
stpcpy          |check|
stpncpy         |check|
strcat          |check|
strchr          |check|
strcmp          |check|
strcpy          |check|
strcspn         |check|
strlcat         |check|
strlcpy         |check|
strlen          |check|
strncat         |check|
strncmp         |check|
strncpy         |check|
strnlen         |check|
strpbrk         |check|
strrchr         |check|
strspn          |check|
strstr          |check|
strtok          |check|
strtok_r        |check|
strdup
strndup
=============   =========
