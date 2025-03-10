Getting Started
===============

Let's fetch the llvm-libc sources and build them.

Install dependencies first:

.. code-block:: sh

  $ sudo apt update
  $ sudo apt install git cmake ninja-build clang gcc-multilib

.. code-block:: sh

  $ git clone --depth=1 git@github.com:llvm/llvm-project.git /tmp/llvm-project
  $ mkdir /tmp/llvm-project/build
  $ cd /tmp/llvm-project/build
  $ cmake ../runtimes -GNinja \
    -DLLVM_ENABLE_RUNTIMES="libc;compiler-rt" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang \
    -DLLVM_LIBC_FULL_BUILD=ON \
    -DLLVM_LIBC_INCLUDE_SCUDO=ON \
    -DCOMPILER_RT_BUILD_SCUDO_STANDALONE_WITH_LLVM_LIBC=ON \
    -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
    -DCOMPILER_RT_SCUDO_STANDALONE_BUILD_SHARED=OFF
  $ ninja libc libm

This will produce the following artifacts:

.. code-block::

  llvm-project/build/libc/lib/libc.a
  llvm-project/build/libc/lib/libm.a
  llvm-project/build/libc/startup/linux/crt1.o
  llvm-project/build/libc/include/**.h

We can then compile and run hello world via:

.. code-block:: c++

  // hello.c
  #include <stdio.h>
  int main () { puts("hello world"); }

.. code-block:: sh

   $ clang -nostdinc -nostdlib hello.c -I libc/include \
     -I $(clang -print-resource-dir)/include libc/startup/linux/crt1.o \
     libc/lib/libc.a
   $ ./a.out
   hello world

This was what we call a "full build" of llvm-libc. From here, you can visit
:ref:`full_host_build` for more info, :ref:`full_cross_build` for cross
compiling, :ref:`overlay_mode` for mixing llvm-libc with another libc, or
:ref:`libc_gpu` for targeting GPUs.
