.. _fullbuild_mode:

==============
Fullbuild Mode
==============

.. contents:: Table of Contents
  :depth: 4
  :local:

The *fullbuild* mode of LLVM's libc is the mode in which it is being used as
the only libc (as opposed to the :ref:`overlay_mode` in which it is used along
with the system libc.) Hence, to start using it that way, you will have to build
and install the ``libc.a`` static archive from LLVM's libc as well as the
start-up objects and public headers provided by it. In this document, we will
present a way to set up a *sysroot* (see the documentation of the ``--sysroot``
option here: `<https://gcc.gnu.org/onlinedocs/gcc/Directory-Options.html>`_)
which includes not only the components of LLVM's libc, but also full a LLVM only
toolchain consisting of the `clang <https://clang.llvm.org/>`_ compiler, the
`lld <https://lld.llvm.org/>`_ linker and the
`compiler-rt <https://compiler-rt.llvm.org/>`_ runtime libraries. LLVM's libc
is not yet complete enough to allow using and linking a C++ application against
a C++ standard library (like libc++). Hence, we do not include a C++ standard
library in the sysroot.

.. note:: When the libc is complete enough, we should be able to include
   `libc++ <https://libcxx.llvm.org/>`_, libcxx-abi and libunwind in the
   toolchain and use them to build and link C++ applications.

Building the full libc
======================

LLVM's libc uses `Scudo <https://llvm.org/docs/ScudoHardenedAllocator.html>`_
as its allocator. So, when building the full libc, we should specify that we
want Scudo to be included in the libc. Since the libc currently only supports
static linking, we also specify that we do not want a shared library for Scudo.
A typical ``cmake`` configure step will look like this:

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build
  $> cd build
  $> cmake ../llvm -G Ninja                                \
     -DLLVM_ENABLE_PROJECTS="clang;libc;lld;compiler-rt"   \
     -DCMAKE_BUILD_TYPE=<Debug|Release>                    \ # Select build type
     -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
     -DLLVM_LIBC_FULL_BUILD=ON      \  # We want the full libc
     -DLLVM_LIBC_INCLUDE_SCUDO=ON   \  # Include Scudo in the libc
     -DCOMPILER_RT_BUILD_SCUDO_STANDALONE_WITH_LLVM_LIBC=ON \
     -DCOMPILER_RT_BUILD_GWP_ASAN=OFF                       \
     -DCOMPILER_RT_SCUDO_STANDALONE_BUILD_SHARED=OFF        \
     -DCMAKE_INSTALL_PREFIX=<SYSROOT>  # Specify a sysroot directory

Since we want to include ``clang``, ``lld`` and ``compiler-rt`` in our
toolchain, we list them in ``LLVM_ENABLE_PROJECTS`` along with ``libc``. The
option ``CMAKE_INSTALL_PREFIX`` specifies the sysroot directory in which to
install the new toolchain.

Installation
============

To build and install the libc, clang (and its support libraries and builtins),
lld and compiler-rt, run the following command after the above ``cmake``
command:

.. code-block:: sh

   $> ninja install-clang install-builtins install-compiler-rt  \
      install-core-resource-headers install-libc install-lld

Once the above command completes successfully, the ``<SYSROOT>`` directory you
have specified with the CMake configure step above will contain a full LLVM-only
toolchain with which you can build practical/real-world C applications. See
`<https://github.com/llvm/llvm-project/tree/main/libc/examples>`_ for examples
of how to start using this new toolchain.

Linux Headers
=============

If you are using the full libc on Linux, then you will also need to install
Linux headers in your sysroot. It is left to the reader to figure out the best
way to install Linux headers on the system they want to use the full libc on.
