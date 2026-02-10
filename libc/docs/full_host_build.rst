.. _full_host_build:

===============
Full Host Build
===============

.. contents:: Table of Contents
   :depth: 1
   :local:

.. note::
   Fullbuild requires running headergen, which is a python program that depends on
   pyyaml. The minimum versions are listed on the :ref:`header_generation`
   page, as well as additional information.

Standard Building and Testing
=============================

.. note::
   If your build fails with an error saying the compiler can't find
   ``<asm/unistd.h>`` or similar then you're probably missing the symlink from
   ``/usr/include/asm`` to ``/usr/include/<HOST TRIPLE>/asm``. Installing the
   ``gcc-multilib`` package creates this symlink, or you can do it manually with
   this command:
   ``sudo ln -s /usr/include/<HOST TRIPLE>/asm /usr/include/asm``
   (your host triple will probably be similar to ``x86_64-linux-gnu``)

For basic development, such as adding new functions or fixing bugs, you can build
and test the libc directly without setting up a full sysroot. This approach
is faster and sufficient for most contributors.

To configure the build, create a build directory and run ``cmake``:

.. code-block:: sh

   cmake \
      -B build \
      -S runtimes \
      -G Ninja \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DLLVM_ENABLE_RUNTIMES="libc;compiler-rt" \
      -DLLVM_LIBC_FULL_BUILD=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      -DLLVM_LIBC_INCLUDE_SCUDO=ON \
      -DCOMPILER_RT_BUILD_SCUDO_STANDALONE_WITH_LLVM_LIBC=ON \
      -DCOMPILER_RT_BUILD_GWP_ASAN=OFF                       \
      -DCOMPILER_RT_SCUDO_STANDALONE_BUILD_SHARED=OFF        \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DLLVM_ENABLE_SPHINX=ON -DLIBC_INCLUDE_DOCS=ON \
      -DLIBC_CMAKE_VERBOSE_LOGGING=ON

After configuring the build, you can build the libc, math library, and run the
tests with the following command:

.. code-block:: sh

   ninja -C build libc libm check-libc

To run a specific unit test for a function, you can target it directly using its
full name:

.. code-block:: sh

   ninja -C build libc.test.src.<HEADER>.<FUNCTION>_test.__unit__

For example, to run the test for ``isalpha`` in ``ctype.h``:

.. code-block:: sh

   ninja -C build libc.test.src.ctype.isalpha_test.__unit__

Building Documentation
======================

If you have Sphinx installed, you can build the libc documentation locally. The
build configuration above already includes the necessary flags
(``-DLLVM_ENABLE_SPHINX=ON -DLIBC_INCLUDE_DOCS=ON``).

To generate the HTML documentation:

.. code-block:: sh

   ninja -C build docs-libc-html

The generated documentation will be available in the ``docs/libc/html`` directory
within your build folder.

Building a Simple Sysroot
=========================

.. warning::
   The LLVM libc is missing many critical functions needed to build non-trivial applications. If you
   are not currently working on porting the libc, we recommend sticking with your system libc. However,
   ignoring warnings like this are how most of us got into this business. So: Speak friend and enter.

This document describes how to set up a simple sysroot and a compiler that uses it from
scratch. These are not full cross-compilation instructions. We make a few
assumptions:

 * The host and target are the same architecture and OS. For example, building a Linux x86-64 libc on a Linux x86-64 host.
 * The host has a working and recent Clang toolchain. Clang 21 has been tested.
 * Your container is using Debian Testing or a derived distribution. Other distributions likely work but the package names and paths may differ.
 * You have root access to your machine to set up the compiler wrapper.

For more comprehensive instructions on setting up a sysroot, see the `official LLVM
guide <https://llvm.org/docs/HowToCrossCompileLLVM.html#setting-up-a-sysroot>`_.


Step 1: Preparation
-------------------

First, set up the environment variables for your sysroot path and the major
version of your host Clang.

.. code-block:: sh

   SYSROOT=$(readlink -f ~/sysroot)

Step 2: Linux Headers
---------------------

Next, install the Linux kernel headers into your sysroot. For this guide, we'll
copy the headers from the host system's ``/usr/include`` directory. This
includes ``linux``, ``asm-generic``, and the architecture-specific ``asm``
headers.

.. code-block:: sh

   # Create the include directory
   mkdir -p $SYSROOT/usr/include

   # Copy the header directories
   cp -R /usr/include/linux $SYSROOT/usr/include/
   cp -R /usr/include/asm-generic $SYSROOT/usr/include/
   # Use -L to dereference the asm symlink and copy the actual files
   cp -R -L /usr/include/asm $SYSROOT/usr/include/

.. note::
   For a more production-ready sysroot, you would typically download a specific
   kernel version and install the headers using ``make headers_install``
   configured for the target architecture and installation path.

Step 3: Build and Install Runtimes
----------------------------------

Now, configure the build for LLVM libc and compiler-rt. We're building with
llvm instead of runtimes because we need to install the
``clang-resource-headers`` that provide ``stdarg.h``, ``stddef.h`` and others.

.. code-block:: sh

   cmake \
      -S llvm \
      -B build-runtimes \
      -G Ninja \
      -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF \
      -DCMAKE_INSTALL_PREFIX=$SYSROOT/usr \
      -DLLVM_ENABLE_PROJECTS="clang"   \
      -DLLVM_ENABLE_RUNTIMES="libc;compiler-rt" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_LIBC_FULL_BUILD=ON \
      -DLIBC_INCLUDE_DOCS=OFF \
      -DLLVM_LIBC_INCLUDE_SCUDO=ON \
      -DCOMPILER_RT_BUILD_SCUDO_STANDALONE_WITH_LLVM_LIBC=ON \
      -DCOMPILER_RT_BUILD_GWP_ASAN=OFF                       \
      -DCOMPILER_RT_SCUDO_STANDALONE_BUILD_SHARED=OFF        \
      -DCOMPILER_RT_BUILD_BUILTINS:BOOL=TRUE \
      -DCOMPILER_RT_BUILD_CRT:BOOL=TRUE \
      -DCOMPILER_RT_BUILD_GWP_ASAN:BOOL=FALSE

After configuring, build and install the necessary components:

.. code-block:: sh

   ninja -C build-runtimes install-clang-resource-headers install-libc install-compiler-rt install-builtins

Step 4: Configure the Compiler Wrapper
--------------------------------------

To make using the new toolchain easier, you can create a Clang configuration
file. This allows you to avoid passing long command line arguments every time
you compile a program.

1. Identify the directory where your Clang binary is located:

.. code-block:: sh

   CLANG_DIR=$(dirname $(readlink -f /usr/bin/clang))

2. Create a symlink to ``clang`` named ``llvm-libc-clang`` in that directory:

.. code-block:: sh

   sudo ln -sf $CLANG_DIR/clang /usr/bin/llvm-libc-clang

3. Create the configuration file in the same directory. Clang automatically looks
   for a file named ``<executable-name>.cfg`` in the same directory as the
   executable. Use the following command to generate it with your environment
   variables:

.. code-block:: sh

   CLANG_VERSION=$(build-runtimes/bin/clang -dumpversion | cut -d. -f1)

   cat <<EOF | sudo tee $CLANG_DIR/llvm-libc-clang.cfg
   --target=x86_64-unknown-linux-llvm
   --sysroot=$SYSROOT
   -resource-dir=$SYSROOT/usr/lib/clang/$CLANG_VERSION
   --rtlib=compiler-rt
   --unwindlib=none
   -static
   EOF

Step 5: Verification
--------------------

You can now use your newly built toolchain by running your wrapper.

.. code-block:: C

   // hello.c
   #include <stdio.h>
   int main() {
      printf("Hello, World!\n");
      return 0;
   }

Compile and run the example:

.. code-block:: sh

   llvm-libc-clang hello.c
   ./a.out

