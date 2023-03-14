.. _full_cross_build:

================
Full Cross Build
================

.. contents:: Table of Contents
   :depth: 1
   :local:

In this document, we will present a recipe to cross build a full libc. When we
say *cross build* a full libc, we mean that we will build the libc for a target
system which is not the same as the system on which the libc is being built.
For example, you could be building for a bare metal aarch64 *target* on a Linux
x86_64 *host*.

Configure the full cross build of the libc
==========================================

Below is a simple recipe to configure the libc for a cross build.  In this,
we've set Ninja as the generator, and are building the full libc.

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build
  $> cd build
  $> cmake ../llvm  \
     -G Ninja \
     -DLLVM_ENABLE_PROJECTS=libc  \
     -DCMAKE_C_COMPILER=clang \
     -DCMAKE_CXX_COMPILER=clang++  \
     -DLLVM_LIBC_FULL_BUILD=ON \
     -DLIBC_TARGET_TRIPLE=<Your target triple>

We will go over the special options passed to the ``cmake`` command above.

* **Enabled Projects** - Since we want to build the libc project, we list
  ``libc`` as the enabled project.
* **The full build option** - Since we want to build the full libc, we pass
  ``-DLLVM_LIBC_FULL_BUILD=ON``.
* **The target triple** - This is the target triple of the target for which
  we are building the libc. For example, for a Linux 32-bit Arm target,
  one can specify it as ``arm-linux-eabi``.

Build and install
=================

After configuring the build with the above ``cmake`` command, one can build the
the libc for the target with the following command:

.. code-block:: sh
   
   $> ninja libc

The above ``ninja`` command will build the ``libc.a`` static archive for the
target specified with ``-DLIBC_TARGET_TRIPLE`` to the ``cmake`` command.

Building for bare metal
=======================

To build for bare metal, all one has to do is to specify the
`system <https://clang.llvm.org/docs/CrossCompilation.html#target-triple>`_
component of the target triple as ``none``. For example, to build for a
32-bit arm target on bare metal, one can use a target triple like
``arm-none-eabi``.
