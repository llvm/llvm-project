.. _getting_started:

===============
Getting Started
===============

This guide provides a single, robust path for new users and contributors to 
build, test, and verify LLVM-libc. We use the **runtimes build** (see 
:ref:`build_concepts` for more information) because it is faster and sufficient 
for most development tasks.

1. Install Dependencies
=======================

To build LLVM-libc, you will need a recent version of Clang (v15+) and basic 
build tools. On a Debian/Ubuntu-based system, you can install these using 
``apt-get``:

.. code-block:: sh

   sudo apt-get update
   sudo apt-get install git cmake ninja-build clang gcc-multilib

2. Clone and Configure
======================

The following command clones the complete LLVM project and configures the 
build for LLVM-libc. We include ``compiler-rt`` to enable the Scudo memory 
allocator.

.. code-block:: sh

   git clone --depth=1 https://github.com/llvm/llvm-project.git
   cd llvm-project
   cmake -G Ninja -S runtimes -B build \
     -DLLVM_ENABLE_RUNTIMES="libc;compiler-rt" \
     -DLLVM_LIBC_FULL_BUILD=ON \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_C_COMPILER=clang \
     -DCMAKE_CXX_COMPILER=clang++ \
     -DLLVM_LIBC_INCLUDE_SCUDO=ON \
     -DCOMPILER_RT_BUILD_SCUDO_STANDALONE_WITH_LLVM_LIBC=ON \
     -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
     -DCOMPILER_RT_SCUDO_STANDALONE_BUILD_SHARED=OFF \
     -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

3. Build and Test
=================

After configuring, you can build the library, the math library (libm), and 
run all unit tests:

.. code-block:: sh

   ninja -C build libc libm check-libc

To run a specific test, such as ``isalpha`` in ``ctype.h``:

.. code-block:: sh

   ninja -C build libc.test.src.ctype.isalpha_test.__unit__

4. Verify with Hello World
==========================

To verify your build, create a simple ``hello.c`` file:

.. code-block:: c

   #include <stdio.h>

   int main() {
     printf("Hello world from LLVM-libc!\n");
     return 0;
   }

Compile it using the build artifacts:

.. code-block:: sh

   clang -nostdinc -nostdlib hello.c -o hello \
     -I build/libc/include \
     -I $(clang -print-resource-dir)/include \
     build/libc/startup/linux/crt1.o \
     build/libc/lib/libc.a \
     build/libc/lib/libm.a

Finally, run the executable:

.. code-block:: sh

   ./hello
   # Output: Hello world from LLVM-libc!

This setup builds LLVM-libc as a standalone library using the 
recommended set of flags. From here, you can visit :ref:`full_host_build` 
for advanced sysroot setup, :ref:`overlay_mode` to learn about using 
LLVM-libc to augment your system's C library, or :ref:`build_concepts` 
to understand other build scenarios.
