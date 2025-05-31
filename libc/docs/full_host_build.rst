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

In this document, we will present a recipe to build the full libc for the host.
When we say *build the libc for the host*, the goal is to build the libc for
the same system on which the libc is being built. First, we will explain how to
build for developing LLVM-libc, then we will explain how to build LLVM-libc as
part of a complete toolchain.

Configure the build for development
===================================


Below is the list of commands for a simple recipe to build LLVM-libc for
development. In this we've set the Ninja generator, set the build type to
"Debug", and enabled the Scudo allocator. This build also enables generating the
documentation and verbose cmake logging, which are useful development features.

.. note::
   if your build fails with an error saying the compiler can't find
   ``<asm/unistd.h>`` or similar then you're probably missing the symlink from
   ``/usr/include/asm`` to ``/usr/include/<HOST TRIPLE>/asm``. Installing the
   ``gcc-multilib`` package creates this symlink, or you can do it manually with
   this command:
   ``sudo ln -s /usr/include/<HOST TRIPLE>/asm /usr/include/asm``
   (your host triple will probably be similar to ``x86_64-linux-gnu``)

.. code-block:: sh

   $> cd llvm-project  # The llvm-project checkout
   $> mkdir build
   $> cd build
   $> cmake ../runtimes \
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

Build and test
==============

After configuring the build with the above ``cmake`` command, one can build test
libc with the following command:

.. code-block:: sh

   $> ninja libc libm check-libc

To build the docs run this command:


.. code-block:: sh

   $> ninja docs-libc-html

To run a specific test, use the following:

.. code-block:: sh

   $> ninja libc.test.src.<HEADER>.<FUNCTION>_test.__unit__
   $> ninja libc.test.src.ctype.isalpha_test.__unit__ # EXAMPLE

Configure the complete toolchain build
======================================

For a complete toolchain we recommend creating a *sysroot* (see the documentation
of the ``--sysroot`` option here:
`<https://gcc.gnu.org/onlinedocs/gcc/Directory-Options.html>`_) which includes
not only the components of LLVM's libc, but also a full LLVM only toolchain
consisting of the `clang <https://clang.llvm.org/>`_ compiler, the
`lld <https://lld.llvm.org/>`_ linker and the
`compiler-rt <https://compiler-rt.llvm.org/>`_ runtime libraries. LLVM-libc is
not quite complete enough to allow using and linking a C++ application against
a C++ standard library (like libc++). Hence, we do not include
`libc++ <https://libcxx.llvm.org/>`_ in the sysroot.

.. note:: When the libc is complete enough, we should be able to include
   `libc++ <https://libcxx.llvm.org/>`_, libcxx-abi and libunwind in the
   LLVM only toolchain and use them to build and link C++ applications.

Below is the cmake command for a bootstrapping build of LLVM. This will build
clang and lld with the current system's toolchain, then build compiler-rt and
LLVM-libc with that freshly built clang. This ensures that LLVM-libc can take
advantage of the latest clang features and optimizations.

This build also uses Ninja as cmake's generator, and sets lld and compiler-rt as
the default for the fresh clang. Those settings are recommended, but the build
should still work without them. The compiler-rt options are required for
building `Scudo <https://llvm.org/docs/ScudoHardenedAllocator.html>`_ as the
allocator for LLVM-libc.

.. note::
   if your build fails with an error saying the compiler can't find
   ``<asm/unistd.h>`` or similar then you're probably missing the symlink from
   ``/usr/include/asm`` to ``/usr/include/<TARGET TRIPLE>/asm``. Installing the
   ``gcc-multilib`` package creates this symlink, or you can do it manually with
   this command:
   ``sudo ln -s /usr/include/<TARGET TRIPLE>/asm /usr/include/asm``

.. code-block:: sh

   $> cd llvm-project  # The llvm-project checkout
   $> mkdir build
   $> cd build
   $> SYSROOT=/path/to/sysroot # Remember to set this!
   $> cmake ../llvm  \
      -G Ninja  \
      -DLLVM_ENABLE_PROJECTS="clang;lld"   \
      -DLLVM_ENABLE_RUNTIMES="libc;compiler-rt" \
      -DCMAKE_BUILD_TYPE=Release  \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DLLVM_LIBC_FULL_BUILD=ON \
      -DLLVM_LIBC_INCLUDE_SCUDO=ON \
      -DCOMPILER_RT_BUILD_SCUDO_STANDALONE_WITH_LLVM_LIBC=ON \
      -DCOMPILER_RT_BUILD_GWP_ASAN=OFF                       \
      -DCOMPILER_RT_SCUDO_STANDALONE_BUILD_SHARED=OFF        \
      -DCLANG_DEFAULT_LINKER=lld \
      -DCLANG_DEFAULT_RTLIB=compiler-rt \
      -DCMAKE_INSTALL_PREFIX=$SYSROOT

Build and install
=================

.. TODO: add this warning to the cmake
.. warning::
   Running these install commands without setting a ``$SYSROOT`` will install
   them into your system include path, which may break your system. If you're
   just trying to develop libc, then just run ``ninja check-libc`` to build the
   libc and run the tests. If you've already accidentally installed the headers,
   you may need to delete them from ``/usr/local/include``.

After configuring the build with the above ``cmake`` command, one can build and
install the toolchain with

.. code-block:: sh

   $> ninja install-clang install-builtins install-compiler-rt  \
      install-core-resource-headers install-libc install-lld

or

.. code-block:: sh

   $> ninja install

Once the above command completes successfully, the ``$SYSROOT`` directory you
have specified with the CMake configure step above will contain a full LLVM-only
toolchain with which you can build practical/real-world C applications. See
`<https://github.com/llvm/llvm-project/tree/main/libc/examples>`_ for examples
of how to start using this new toolchain.

Linux Headers
=============

If you are using the full libc on Linux, then you will also need to install
Linux headers in your sysroot.  Let's build them from source.

.. code-block:: sh

   $> git clone --depth=1 git://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git /tmp/linux
   $> make LLVM=1 INSTALL_HDR_PATH=/path/to/sysroot -C /tmp/linux headers_install

The headers can be built to target non-host architectures by adding the
``ARCH={arm|arm64|i386}`` to the above invocation of ``make``.

Using your newly built libc
===========================

You can now use your newly built libc nearly like you would use any compiler
invocation:

.. code-block:: sh

   $> /path/to/sysroot/bin/clang -static main.c

.. warning::
   Because the libc does not yet support dynamic linking, the -static parameter
   must be added to all clang invocations.


You can make sure you're using the newly built toolchain by trying out features
that aren't yet supported by the system toolchain, such as fixed point. The
following is an example program that demonstrates the difference:

.. code-block:: C

   // $ $SYSROOT/bin/clang example.c -static -ffixed-point --sysroot=$SYSROOT

   #include <stdio.h>
   int main() {
      printf("Hello, World!\n%.9f\n%.9lK\n",
         4294967295.000000001,
         4294967295.000000001ulK);
      return 0;
   }

