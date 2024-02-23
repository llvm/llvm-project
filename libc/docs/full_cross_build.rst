.. _full_cross_build:

================
Full Cross Build
================

.. contents:: Table of Contents
   :depth: 1
   :local:

In this document, we will present recipes to cross build the full libc. When we
say *cross build* a full libc, we mean that we will build the full libc for a
target system which is not the same as the system on which the libc is being
built. For example, you could be building for a bare metal aarch64 *target* on a
Linux x86_64 *host*.

There are three main recipes to cross build the full libc. Each one serves a
different use case. Below is a short description of these recipes to help users
pick the recipe that best suites their needs and contexts.

* **Standalone cross build** - Using this recipe one can build the libc using a
  compiler of their choice. One should use this recipe if their compiler can
  build for the host as well as the target.
* **Runtimes cross build** - In this recipe, one will have to first build the
  libc build tools for the host separately and then use those build tools to
  build the libc. Users can use the compiler of their choice to build the
  libc build tools as well as the libc. One should use this recipe if they
  have to use a host compiler to build the build tools for the host and then
  use a target compiler (which is different from the host compiler) to build
  the libc.
* **Bootstrap cross build** - In this recipe, one will build the ``clang``
  compiler and the libc build tools for the host first, and then use them to
  build the libc for the target. Unlike with the runtimes build recipe, the
  user does not have explicitly build ``clang`` and other libc build tools.
  They get built automatically before building the libc. One should use this
  recipe if they intend use the built ``clang`` and the libc as part of their
  toolchain for the target.

The following sections present the three recipes in detail.

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
  $> cmake ../llvm  \
     -G Ninja \
     -DLLVM_ENABLE_PROJECTS=libc  \
     -DCMAKE_C_COMPILER=$C_COMPILER \
     -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
     -DLLVM_LIBC_FULL_BUILD=ON \
     -DLIBC_TARGET_TRIPLE=<Your target triple> \
     -DCMAKE_BUILD_TYPE=<Release|Debug>

We will go over the special options passed to the ``cmake`` command above.

* **Enabled Projects** - Since we want to build the libc project, we list
  ``libc`` as the enabled project.
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

Runtimes cross build
====================

The *runtimes cross build* is very similar to the standalone crossbuild but the
user will have to first build the libc build tools for the host separately. One
should use this recipe if they want to use a different host and target compiler.
Note that the libc build tools MUST be in sync with the libc. That is, the
libc build tools and the libc, both should be built from the same source
revision. At the time of this writing, there is only one libc build tool that
has to be built separately. It is done as follows:

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build-libc-tools # A different build directory for the build tools
  $> cd build-libc-tools
  $> HOST_C_COMPILER=<C compiler for the host> # For example "clang"
  $> HOST_CXX_COMPILER=<C++ compiler for the host> # For example "clang++"
  $> cmake ../llvm  \
     -G Ninja \
     -DLLVM_ENABLE_PROJECTS=libc  \
     -DCMAKE_C_COMPILER=$HOST_C_COMPILER \
     -DCMAKE_CXX_COMPILER=$HOST_CXX_COMPILER  \
     -DLLVM_LIBC_FULL_BUILD=ON \
     -DCMAKE_BUILD_TYPE=Debug # User can choose to use "Release" build type
  $> ninja libc-hdrgen

The above commands should build a binary named ``libc-hdrgen``. Copy this binary
to a directory of your choice.

CMake configure step
--------------------

After copying the ``libc-hdrgen`` binary to say ``/path/to/libc-hdrgen``,
configure the libc build using the following command:

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build
  $> cd build
  $> TARGET_C_COMPILER=<C compiler for the target>
  $> TARGET_CXX_COMPILER=<C++ compiler for the target>
  $> HDRGEN=</path/to/libc-hdrgen>
  $> TARGET_TRIPLE=<Your target triple>
  $> cmake ../runtimes  \
     -G Ninja \
     -DLLVM_ENABLE_RUNTIMES=libc  \
     -DCMAKE_C_COMPILER=$TARGET_C_COMPILER \
     -DCMAKE_CXX_COMPILER=$TARGET_CXX_COMPILER \
     -DLLVM_LIBC_FULL_BUILD=ON \
     -DLIBC_HDRGEN_EXE=$HDRGEN \
     -DLIBC_TARGET_TRIPLE=$TARGET_TRIPLE \
     -DCMAKE_BUILD_TYPE=Debug # User can choose to use "Release" build type

Note the differences in the above cmake command versus the one used in the
CMake configure step of the standalone build recipe:

* Instead of listing ``libc`` in ``LLVM_ENABLED_PROJECTS``, we list it in
  ``LLVM_ENABLED_RUNTIMES``.
* Instead of using ``llvm-project/llvm`` as the root CMake source directory,
  we use ``llvm-project/runtimes`` as the root CMake source directory.
* The path to the ``libc-hdrgen`` binary built earlier is specified with
  ``-DLIBC_HDRGEN_EXE=/path/to/libc-hdrgen``.

Build step
----------

The build step in the runtimes build recipe is exactly the same as that of
the standalone build recipe:

.. code-block:: sh

    $> ninja libc libm

As with the standalone build recipe, the above ninja command will build the
libc static archives for the target specified with ``-DLIBC_TARGET_TRIPLE`` in
the CMake configure step.


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

Note how the above cmake command differs from the one used in the other two
recipes:

* ``clang`` is listed in ``-DLLVM_ENABLE_PROJECTS`` and ``libc`` is
  listed in ``-DLLVM_ENABLE_RUNTIMES``.
* The CMake root source directory is ``llvm-project/llvm``.
* The target triple is specified with ``-DLLVM_RUNTIME_TARGETS``.

Build step
----------

The build step is similar to the other two recipes:

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
