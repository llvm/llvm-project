.. _build_concepts:

==============
Build Concepts
==============

Most people don't need to build their own C library — the one provided by their
system works well. However, LLVM-libc's **Overlay Mode** can provide key updates
like faster or more consistent math functions for projects that need them.

For those who do need a full C library, LLVM-libc supports several build
configurations depending on your target environment and intended usage.

The Five Build Scenarios
========================

1. Overlay Mode (Incremental Adoption)
--------------------------------------

In Overlay Mode, LLVM-libc functions are compiled alongside the host's existing 
system library (like ``glibc``). Only the functions explicitly implemented in 
LLVM-libc are used; the rest "fall back" to the system library. This is the 
preferred method for most contributors as it is the fastest to build and test.

To configure for an overlay build, point CMake to the ``runtimes`` directory 
and set ``LLVM_LIBC_FULL_BUILD=OFF`` (which is the default). This will build a 
static archive named ``libllvmlibc.a``:

.. code-block:: sh

   cmake -S runtimes -B build -DLLVM_ENABLE_RUNTIMES="libc" \
         -DLLVM_LIBC_FULL_BUILD=OFF ...

2. Full Build Mode (Standalone Library)
---------------------------------------

In Full Build Mode, LLVM-libc is a complete replacement for the system library. 
This is used to build standalone ``libc.a`` and ``libm.a`` (with separate CMake 
targets) for a new operating system or to generate a sysroot for a specific target.

To configure for a full build, set ``LLVM_LIBC_FULL_BUILD=ON``:

.. code-block:: sh

   cmake -S runtimes -B build -DLLVM_ENABLE_RUNTIMES="libc;compiler-rt" \
         -DLLVM_LIBC_FULL_BUILD=ON ...

3. Bootstrap Build
------------------

A bootstrap build first builds the compiler (Clang) and other LLVM tools using
the host compiler, and then uses that newly-built Clang to build the libc.
This ensures you are using a matched toolchain where the compiler and
the library are built for each other.

To configure a bootstrap build, you point CMake to the ``llvm`` directory:

.. code-block:: sh

   cmake -S llvm -B build -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_ENABLE_RUNTIMES="libc;compiler-rt" ...

4. Cross-compiler Build (Targeting Other Architectures)
-------------------------------------------------------

Used when you want to build LLVM-libc for a different architecture than you are 
currently running on (e.g., building on x86_64 for an aarch64 target). 
This requires a cross-compiler or a toolchain file.

5. Bootstrap Cross-compiler (New Environment)
---------------------------------------------

For users who are starting from scratch (e.g., with only Linux kernel headers) 
and want to generate a full C compiler and sysroot for their target. This is 
the most common path for those building entire environments to tinker in.


