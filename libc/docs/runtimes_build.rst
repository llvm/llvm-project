Building libc using the runtimes build setup
============================================

The runtimes build of the LLVM toolchain first builds clang and then builds the
various runtimes (like ``libc++`` and ``compiler-rt``) and LLVM binutils (like
``llvm-objcopy`` and ``llvm-readelf``) using the freshly built clang. One can
build libc also as in the same manner. As of this writing, only the ABI agnostic
parts of the libc are included when built in that manner. This allows interested
users to continue using their system libc's headers while linking to LLVM libc's
implementations when they are available. To build libc using the runtimes build
setup, one needs to include the ``libc`` project in the list of the enabled
runtimes when configuring the build:

.. code-block:: shell

   $> cmake ../llvm -GNinja -DLLVM_ENABLE_PROJECTS="llvm;clang" \
      -DLLVM_ENABLE_RUNTIMES=libc

Note that Ninja is used as the generator in the above CMake command. Hence, to
actually build the libc, one has to build the Ninja target named ``llvmlibc``:

.. code-block:: shell

   $> ninja llvmlibc

If a different generator is used, then the build command should be suitably
adapted to build the target ``llvmlibc``. Building that target will produce a
static archive which includes all ABI agnostic functions available in LLVM libc.

Future direction
----------------

We plan to enhance the runtimes build of LLVM libc to include ABI sensitive
parts and to also generate the public headers. Likewise, we would like to
provide an option to build other runtimes like ``libc++`` and ``compiler-rt``
against LLVM libc.
