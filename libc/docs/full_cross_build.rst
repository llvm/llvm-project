.. _full_cross_build:

================
Full Cross Build
================

.. contents:: Table of Contents
   :depth: 1
   :local:

.. note:: 
   Fullbuild requires running headergen, which is a python program that depends on
   pyyaml. The minimum versions are listed on the :ref:`header_generation`
   page, as well as additional information.

In this document, we will present recipes to cross build the full libc. When we
say *cross build* a full libc, we mean that we will build the full libc for a
target system which is not the same as the system on which the libc is being
built. For example, you could be building for a bare metal aarch64 *target* on a
Linux x86_64 *host*.

There are two main recipes to cross build the full libc. Each one serves a
different use case. Below is a short description of these recipes to help users
pick the recipe that best suites their needs and contexts.

* **Standalone cross build** - Using this recipe one can build the libc using a
  compiler of their choice. One should use this recipe if their compiler can
  build for the host as well as the target.
* **Bootstrap cross build** - In this recipe, one will build the ``clang``
  compiler and the libc build tools for the host first, and then use them to
  build the libc for the target. Unlike with the standalone build recipe, the
  user does not have explicitly build ``clang`` and other build tools.
  They get built automatically before building the libc. One should use this
  recipe if they intend use the built ``clang`` and the libc as part of their
  toolchain for the target.

The following sections present the two recipes in detail.

Standalone cross build
======================

In the *standalone crossbuild* recipe, the system compiler or a custom compiler
of user's choice is used to build the libc. The necessary build tools for the
host are built first, and those build tools are then used to build the libc for
the target. Both these steps happen automatically, as in, the user does not have
to explicitly build the build tools first and then build the libc. A point to
keep in mind is that the compiler used should be capable of building for the
host as well as the target.

CMake configure step
--------------------

Below is the CMake command to configure the standalone crossbuild of the libc.

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build
  $> cd build
  $> C_COMPILER=<C compiler> # For example "clang"
  $> CXX_COMPILER=<C++ compiler> # For example "clang++"
  $> cmake ../runtimes  \
     -G Ninja \
     -DLLVM_ENABLE_RUNTIMES=libc  \
     -DCMAKE_C_COMPILER=$C_COMPILER \
     -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
     -DLLVM_LIBC_FULL_BUILD=ON \
     -DLIBC_TARGET_TRIPLE=<Your target triple> \
     -DCMAKE_BUILD_TYPE=<Release|Debug>

We will go over the special options passed to the ``cmake`` command above.

* **Enabled Runtimes** - Since we want to build LLVM-libc, we list
  ``libc`` as the enabled runtime.
* **The full build option** - Since we want to build the full libc, we pass
  ``-DLLVM_LIBC_FULL_BUILD=ON``.
* **The target triple** - This is the target triple of the target for which
  we are building the libc. For example, for a Linux 32-bit Arm target,
  one can specify it as ``arm-linux-eabi``.

Build step
----------

After configuring the build with the above ``cmake`` command, one can build the
the libc for the target with the following command:

.. code-block:: sh

   $> ninja libc libm

The above ``ninja`` command will build the libc static archives ``libc.a`` and
``libm.a`` for the target specified with ``-DLIBC_TARGET_TRIPLE`` in the CMake
configure step.

Bootstrap cross build
=====================

In this recipe, the clang compiler and the ``libc-hdrgen`` binary, both are
built automatically before building the libc for the target.

CMake configure step
--------------------

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build
  $> cd build
  $> C_COMPILER=<C compiler> # For example "clang"
  $> CXX_COMPILER=<C++ compiler> # For example "clang++"
  $> TARGET_TRIPLE=<Your target triple>
  $> cmake ../llvm \
     -G Ninja \
     -DCMAKE_C_COMPILER=$C_COMPILER \
     -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
     -DLLVM_ENABLE_PROJECTS=clang \
     -DLLVM_ENABLE_RUNTIMES=libc \
     -DLLVM_LIBC_FULL_BUILD=ON \
     -DLLVM_RUNTIME_TARGETS=$TARGET_TRIPLE \
     -DCMAKE_BUILD_TYPE=Debug

Note how the above cmake command differs from the one used in the other recipe:

* ``clang`` is listed in ``-DLLVM_ENABLE_PROJECTS`` and ``libc`` is
  listed in ``-DLLVM_ENABLE_RUNTIMES``.
* The CMake root source directory is ``llvm-project/llvm``.
* The target triple is specified with ``-DLLVM_RUNTIME_TARGETS``.

Build step
----------

The build step is similar to the other recipe:

.. code-block:: sh

  $> ninja libc

The above ninja command should build the libc static archives for the target
specified with ``-DLLVM_RUNTIME_TARGETS``.

Building for bare metal
=======================

To build for bare metal, all one has to do is to specify the
`system <https://clang.llvm.org/docs/CrossCompilation.html#target-triple>`_
component of the target triple as ``none``. For example, to build for a
32-bit arm target on bare metal, one can use a target triple like
``arm-none-eabi``. Other than that, the libc for a bare metal target can be
built using any of the three recipes described above.

Building for the GPU
====================

To build for a GPU architecture, it should only be necessary to specify the 
target triple as one of the supported GPU targets. Currently, this is either 
``nvptx64-nvidia-cuda`` for NVIDIA GPUs or ``amdgcn-amd-amdhsa`` for AMD GPUs. 
More detailed information is provided in the :ref:`GPU 
documentation<libc_gpu_building>`.
