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

Building the static archive with libc as a normal LLVM project
--------------------------------------------------------------

We can treat the ``libc`` project as any other normal LLVM project and perform
the CMake configure step as follows:

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build
  $> cd build
  $> cmake ../llvm -G Ninja -DLLVM_ENABLE_PROJECTS=”libc”  \
     -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
     -DCMAKE_BUILD_TYPE=<Debug|Release>                    \  # Select build type
     -DCMAKE_INSTALL_PREFIX=<Your prefix of choice>           # Optional

Next, build the libc:

.. code-block:: sh

  $> ninja llvmlibc

The build step will build the static archive the in the directory
``build/projects/libc/lib``. Notice that the above CMake configure step also
specified an install prefix. This is optional, but if one uses it, then they
can follow up the build step with an install step:

.. code-block:: sh

  $> ninja install-llvmlibc

Building the static archive as part of the bootstrap build
----------------------------------------------------------

The bootstrap build is a build mode in which runtime components like libc++,
libcxx-abi, libc etc. are built using the ToT clang. The idea is that this build
produces an in-sync toolchain of compiler + runtime libraries. Such a synchrony
is not essential for the libc but can one still build the overlay static archive
as part of the bootstrap build if one wants to. The first step is to configure
appropriately:

.. code-block:: sh

  $> cmake ../llvm -G Ninja -DLLVM_ENABLE_PROJECTS=”clang” \
     -DLLVM_ENABLE_RUNTIMES=”libc”  \  # libc is listed as runtime and not as a project
     -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
     -DCMAKE_BUILD_TYPE=<Debug|Release>                    \  # Select build type
     -DCMAKE_INSTALL_PREFIX=<Your prefix of choice>           # Optional

The build and install steps are similar to the those used when configured
as a normal project. Note that the build step takes much longer this time
as ``clang`` will be built before building ``libllvmlibc.a``.

.. code-block:: sh

  $> ninja llvmlibc
  $> ninja install-llvmlibc

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
install step or explicity passing any special linker flags/options. One can
simply add ``llvmlibc`` as a link library to that binary's target. For example,
if you want to link ``libllvmlibc.a`` to ``llvm-objcopy``, all you have to do
is to add a CMake command as follows:

.. code-block:: cmake

  target_link_libraries(llvm-objcopy PRIVATE llvmlibc)


