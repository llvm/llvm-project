.. _overlay_mode:

============
Overlay Mode
============

.. contents:: Table of Contents
  :depth: 4
  :local:

One can choose to use LLVM's libc in the overlay mode. In this mode, the link
order semantics are exploited to pick symbols from ``libllvmlibc.a`` (if they
are available in ``libllvmlibc.a``) and the rest are picked from the system
libc. The user programs also have to use header files from the system libc.
Naturally, only functions which do not depend on implementation specific ABI
are included in ``libllvmlibc.a``. Examples of such functions are ``strlen``
and ``round``. Functions like ``fopen`` and friends are not included as they
depend on the implementation specific definition of the ``FILE`` data structure.

Building the libc in the overlay mode
=====================================

There are two different ways in which the libc can be built for use in the
overlay mode. In both the ways, we build a static archive named
``libllvmlibc.a``. We use a rather verbose name with a repeated ``lib`` to make
it clear that it is not the system libc, which is typically named ``libc.a``.
Also, if users choose to mix more than one libc with the system libc, then
the name ``libllvmlibc.a`` makes it absolutely clear that it is the static
archive of LLVM's libc.

Building LLVM-libc as a standalone runtime
------------------------------------------

We can treat the ``libc`` project like any other normal LLVM runtime library by
building it with the following cmake command:

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build
  $> cd build
  $> cmake ../runtimes -G Ninja -DLLVM_ENABLE_RUNTIMES="libc"  \
     -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
     -DCMAKE_BUILD_TYPE=<Debug|Release>                    \  # Select build type
     -DCMAKE_INSTALL_PREFIX=<Your prefix of choice>           # Optional

Next, build the libc:

.. code-block:: sh

  $> ninja libc

Then, run the tests:

.. code-block:: sh

  $> ninja check-libc

The build step will build the static archive the in the directory
``build/projects/libc/lib``. Notice that the above CMake configure step also
specified an install prefix. This is optional, but it's used, then the following
command will install the static archive to the install path:

.. code-block:: sh

  $> ninja install-libc

Building the static archive as part of the bootstrap build
----------------------------------------------------------

The bootstrap build is a build mode in which runtime components like libc++,
libcxx-abi, libc etc. are built using the ToT clang. The idea is that this build
produces an in-sync toolchain of compiler + runtime libraries. This ensures that
LLVM-libc has access to the latest clang features, which should provide the best
performance possible.

.. code-block:: sh

  $> cmake ../llvm -G Ninja -DLLVM_ENABLE_PROJECTS="clang" \
     -DLLVM_ENABLE_RUNTIMES="libc"  \  # libc is listed as runtime and not as a project
     -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
     -DCMAKE_BUILD_TYPE=<Debug|Release>                    \  # Select build type
     -DCMAKE_INSTALL_PREFIX=<Your prefix of choice>           # Optional

The build and install steps are the same as above, but the build step will take
much longer since ``clang`` will be built before building ``libllvmlibc.a``.

.. code-block:: sh

  $> ninja libc
  $> ninja check-libc

Using the overlay static archive
================================

Once built (and optionally installed), the overlay static archive can be linked
to your binaries like any other static archive. For example, when building with
``clang`` on Linux, one should follow a recipe like:


.. code-block:: sh

  $> clang <other compiler and/or linker options> <file.o|c(pp)>     \
     -L <path to the directory in which libllvmlibc.a is installed>  \ # Optional
     -lllvmlibc

If you installed ``libllvmlibc.a`` in a standard linker lookup path, for example
``/usr/local/lib`` on Linux like systems, then specifying the path to the
static archive using the ``-L`` option is not necessary.

Linking the static archive to other LLVM binaries
-------------------------------------------------

Since the libc and other LLVM binaries are developed in the same source tree,
linking ``libllvmlibc.a`` to those LLVM binaries does not require any special
install step or explicitly passing any special linker flags/options. One can
simply add ``llvmlibc`` as a link library to that binary's target. For example,
if you want to link ``libllvmlibc.a`` to ``llvm-objcopy``, all you have to do
is to add a CMake command as follows:

.. code-block:: cmake

  target_link_libraries(llvm-objcopy PRIVATE llvmlibc)


