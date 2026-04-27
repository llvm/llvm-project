.. _full_cross_build:

================
Full Cross Build
================

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

.. note::
   Even though the LLVM libc provides its own complete C library implementation,
   compiling it for a Linux target still requires the Linux kernel API headers for
   that architecture. On Debian-based systems, these and other standard cross-compilation 
   runtimes (like ``libgcc``) can be installed via packages like 
   ``gcc-riscv64-linux-gnu`` and ``linux-libc-dev-riscv64-cross`` (or similar for 
   other architectures). You will need to point CMake to the kernel headers using 
   ``-DLIBC_KERNEL_HEADERS`` (e.g., 
   ``-DLIBC_KERNEL_HEADERS=/usr/riscv64-linux-gnu/include``) so the libc build
   can find headers like ``asm/unistd.h``.

CMake configure step
--------------------

First, set up the environment variables for your compiler and target:

.. code-block:: sh

  C_COMPILER=clang
  CXX_COMPILER=clang++
  TARGET_TRIPLE=aarch64-linux-gnu

Below is the CMake command to configure the standalone crossbuild of the libc.

.. code-block:: sh

  cmake \
     -B build \
     -S runtimes \
     -G Ninja \
     -DLLVM_ENABLE_RUNTIMES=libc  \
     -DCMAKE_C_COMPILER=$C_COMPILER \
     -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
     -DCMAKE_C_COMPILER_TARGET=$TARGET_TRIPLE \
     -DCMAKE_CXX_COMPILER_TARGET=$TARGET_TRIPLE \
     -DLLVM_LIBC_FULL_BUILD=ON \
     -DLIBC_TARGET_TRIPLE=$TARGET_TRIPLE \
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

   ninja -C build libc libm

The above ``ninja`` command will build the libc static archives ``libc.a`` and
``libm.a`` for the target specified with ``-DLIBC_TARGET_TRIPLE`` in the CMake
configure step.

Bootstrap cross build
=====================

In this recipe, the clang compiler is built automatically before building
the libc for the target.

CMake configure step
--------------------

First, set up the environment variables for your compiler and target:

.. code-block:: sh

  C_COMPILER=clang
  CXX_COMPILER=clang++
  TARGET_TRIPLE=aarch64-linux-gnu

Then, configure the CMake build for the bootstrap build:

.. code-block:: sh

  cmake \
     -B build \
     -S llvm \
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

  ninja -C build libc

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

Building and Testing with an Emulator
=====================================

If you are cross-compiling the libc for a different architecture, you can use an emulator
such as QEMU to run the tests. For instance, to cross-compile for ``riscv64`` and run tests
using ``qemu-riscv64``, you can use the standalone cross build recipe with a few additional CMake flags.

CMake configure step
--------------------

Assuming your system compiler (e.g., ``clang++``) supports the RISC-V target,
you can configure the build as follows:

.. code-block:: sh

  cmake \
     -B build \
     -S runtimes \
     -G Ninja \
     -DLLVM_ENABLE_RUNTIMES=libc  \
     -DCMAKE_C_COMPILER=clang \
     -DCMAKE_CXX_COMPILER=clang++ \
     -DCMAKE_C_COMPILER_TARGET=riscv64-linux-gnu \
     -DCMAKE_CXX_COMPILER_TARGET=riscv64-linux-gnu \
     -DLLVM_LIBC_FULL_BUILD=ON \
     -DLIBC_TARGET_TRIPLE=riscv64-linux-gnu \
     -DLIBC_KERNEL_HEADERS=/usr/riscv64-linux-gnu/include \
     -DCMAKE_CROSSCOMPILING_EMULATOR=qemu-riscv64 \
     -DLLVM_ENABLE_LLD=ON \
     -DCMAKE_BUILD_TYPE=Debug

The notable additions are:

* **The compiler target** - We set ``-DCMAKE_C_COMPILER_TARGET=riscv64-linux-gnu`` and ``-DCMAKE_CXX_COMPILER_TARGET=riscv64-linux-gnu`` to tell CMake to pass the correct ``--target`` flags to ``clang`` so that it cross-compiles rather than building for the host.
* **The target triple** - We set ``-DLIBC_TARGET_TRIPLE=riscv64-linux-gnu``
* **The kernel headers** - We set ``-DLIBC_KERNEL_HEADERS=/usr/riscv64-linux-gnu/include``
  to point to the target's Linux API headers (e.g. for ``asm/unistd.h``).
* **The cross-compiling emulator** - We set ``-DCMAKE_CROSSCOMPILING_EMULATOR=qemu-riscv64``
  to tell CMake how to execute the compiled unittests. Note that this requires that
  ``qemu-riscv64`` is installed and available in your ``$PATH``.
* **LLD Linker** - We set ``-DLLVM_ENABLE_LLD=ON`` to ensure the test suite is linked using ``lld``, which is necessary for cross-compilation.

Build and Test step
-------------------

You can then build the libc using:

.. code-block:: sh

  ninja -C build libc

To run the tests for the cross-compiled libc, you must use the hermetic test
suite, which is entirely self-hosted.

.. code-block:: sh

  ninja -C build libc-hermetic-tests

.. note::
   The standard ``check-libc`` target relies on the target's system C++ and C library
   headers. Because these tests aren't hermetic, they are not expected to work for
   a standalone cross-compilation build.
