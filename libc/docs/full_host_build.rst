.. _full_host_build:

===============
Full Host Build
===============

.. contents:: Table of Contents
   :depth: 1
   :local:

In this document, we will present a recipe to build the full libc for the host.
When we say *build the libc for the host*, the goal is to build the libc for
the same system on which the libc is being built. Also, we will take this
opportunity to demonstrate how one can set up a *sysroot* (see the documentation
of the ``--sysroot`` option here:
`<https://gcc.gnu.org/onlinedocs/gcc/Directory-Options.html>`_) which includes
not only the components of LLVM's libc, but also a full LLVM only toolchain
consisting of the `clang <https://clang.llvm.org/>`_ compiler, the
`lld <https://lld.llvm.org/>`_ linker and the
`compiler-rt <https://compiler-rt.llvm.org/>`_ runtime libraries. LLVM's libc is
not yet complete enough to allow using and linking a C++ application against
a C++ standard library (like libc++). Hence, we do not include
`libc++ <https://libcxx.llvm.org/>`_ in the sysroot.

.. note:: When the libc is complete enough, we should be able to include
   `libc++ <https://libcxx.llvm.org/>`_, libcxx-abi and libunwind in the
   LLVM only toolchain and use them to build and link C++ applications.

Configure the full libc build
===============================

Below is the list of commands for a simple recipe to build and install the
libc components along with other components of an LLVM only toolchain.  In this
we've set the Ninja generator, enabled a full compiler suite, set the build
type to "Debug", and enabled the Scudo allocator.  The build also tells clang
to use the freshly built lld and compiler-rt.

.. code-block:: sh

   $> cd llvm-project  # The llvm-project checkout
   $> mkdir build
   $> cd build
   $> SYSROOT=/path/to/sysroot # Remember to set this!
   $> cmake ../llvm  \
      -G Ninja  \
      -DLLVM_ENABLE_PROJECTS="clang;libc;lld;compiler-rt"   \
      -DCMAKE_BUILD_TYPE=Debug  \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DLLVM_LIBC_FULL_BUILD=ON \
      -DLLVM_LIBC_INCLUDE_SCUDO=ON \
      -DCOMPILER_RT_BUILD_SCUDO_STANDALONE_WITH_LLVM_LIBC=ON \
      -DCOMPILER_RT_BUILD_GWP_ASAN=OFF                       \
      -DCOMPILER_RT_SCUDO_STANDALONE_BUILD_SHARED=OFF        \
      -DCLANG_DEFAULT_LINKER=lld \
      -DCLANG_DEFAULT_RTLIB=compiler-rt \
      -DDEFAULT_SYSROOT=$SYSROOT \
      -DCMAKE_INSTALL_PREFIX=$SYSROOT

We will go over some of the special options passed to the ``cmake`` command
above.

* **Enabled Projects** - Since we want to build and install clang, lld
  and compiler-rt along with the libc, we specify
  ``clang;libc;lld;compiler-rt`` as the list of enabled projects.
* **The full build option** - Since we want to do build the full libc, we pass
  ``-DLLVM_LIBC_FULL_BUILD=ON``.
* **Scudo related options** - LLVM's libc uses
  `Scudo <https://llvm.org/docs/ScudoHardenedAllocator.html>`_ as its allocator.
  So, when building the full libc, we should specify that we want to include
  Scudo in the libc. Since the libc currently only supports static linking, we
  also specify that we do not want to build the Scudo shared library.
* **Default sysroot and install prefix** - This is the path to the tool chain
  install directory.  This is the directory where you intend to set up the sysroot.

Build and install
=================

After configuring the build with the above ``cmake`` command, one can build and
install the libc, clang (and its support libraries and builtins), lld and
compiler-rt, with the following command:

.. code-block:: sh

   $> ninja install-clang install-builtins install-compiler-rt  \
      install-core-resource-headers install-libc install-lld

Once the above command completes successfully, the ``$SYSROOT`` directory you
have specified with the CMake configure step above will contain a full LLVM-only
toolchain with which you can build practical/real-world C applications. See
`<https://github.com/llvm/llvm-project/tree/main/libc/examples>`_ for examples
of how to start using this new toolchain.

Linux Headers
=============

If you are using the full libc on Linux, then you will also need to install
Linux headers in your sysroot.  The way to do this varies per system.

These instructions should work on a Debian-based x86_64 system:

.. code-block:: sh

   $> apt download linux-libc-dev
   $> dpkg -x linux-libc-dev*deb .
   $> mv usr/include/* /path/to/sysroot/include
   $> rm -rf usr linux-libc-dev*deb
   $> ln -s x86_64-linux-gnu/asm ~/Programming/sysroot/include/asm

Using your newly built libc
===========================

You can now use your newly built libc nearly like you would use any compiler
invocation:

.. code-block:: sh

   $> /path/to/sysroot/bin/clang -static main.c

.. warning::
   Because the libc does not yet support dynamic linking, the -static parameter
   must be added to all clang invocations.

