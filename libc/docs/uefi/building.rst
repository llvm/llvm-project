.. _libc_uefi_building:

======================
Building libc for UEFI
======================

.. contents:: Table of Contents
  :depth: 4
  :local:

Building LLVM libc for UEFI
===========================

This document will present recipes to build the LLVM C library for UEFI.
UEFI builds use the same :ref:`cross build<full_cross_build>` support as
the other targets. However, the UEFI target has the restriction that it *must*
be built with an up-to-date ``clang`` compiler. This is because UEFI support
in ``clang`` is still an experimental feature.

Currently, it is only possible to build LLVM libc for UEFI for ``x86_64``
CPUs. This is due to the target not being enabled for ``aarch64`` and
``riscv64``.

Once you have finished building, refer to :ref:`libc_uefi_usage` to get started
with the newly built C library.

Standard runtimes build
-----------------------

The simplest way to build for UEFI is to use the existing LLVM runtimes
support. This will automatically handle bootstrapping an up-to-date ``clang``
compiler and use it to build the C library. The following CMake invocation
will instruct it to build the ``libc`` runtime targeting ``x86_64`` CPUs.

.. code-block:: sh

   $> cd llvm-project  # The llvm-project checkout
   $> mkdir build
   $> cd build
   $> cmake ../llvm -G Ninja                                                 \
      -DLLVM_ENABLE_PROJECTS="clang;lld"                                     \
      -DCMAKE_BUILD_TYPE=<Debug|Release>  \ # Select build type
      -DCMAKE_INSTALL_PREFIX=<PATH>       \ # Where the libraries will live
      -DLLVM_DEFAULT_TARGET_TRIPLE=x86_64-unknown-uefi-llvm                  \
      -DLLVM_RUNTIME_TARGETS=x86_64-unknown-uefi-llvm                        \
      -DRUNTIMES_x86_64-unknown-uefi-llvm_LLVM_ENABLE_RUNTIMES=libc          \
      -DRUNTIMES_x86_64-unknown-uefi-llvm_LLVM_LIBC_FULL_BUILD=true          \
   $> ninja install

We need ``clang`` to build the UEFI C library and ``lld`` to link UEFI PE
executables, so we enable them in ``LLVM_ENABLE_PROJECTS``. We then set
``RUNTIMES_<triple>_LLVM_ENABLE_RUNTIMES`` to enable ``libc`` for the UEFI
targets.
