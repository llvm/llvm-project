.. _libc_gpu_building:

======================
Building libs for GPUs
======================

.. contents:: Table of Contents
  :depth: 4
  :local:

Building the GPU C library
==========================

This document will present recipes to build the LLVM C library targeting a GPU
architecture. The GPU build uses the same :ref:`cross build<full_cross_build>`
support as the other targets. However, the GPU target has the restriction that
it *must* be built with an up-to-date ``clang`` compiler. This is because the
GPU target uses several compiler extensions to target GPU architectures.

The LLVM C library currently supports two GPU targets. This is either
``nvptx64-nvidia-cuda`` for NVIDIA GPUs or ``amdgcn-amd-amdhsa`` for AMD GPUs.
Targeting these architectures is done through ``clang``'s cross-compiling
support using the ``--target=<triple>`` flag. The following sections will
describe how to build the GPU support specifically.

Once you have finished building, refer to :ref:`libc_gpu_usage` to get started
with the newly built C library.

Standard runtimes build
-----------------------

The simplest way to build the GPU libc is to use the existing LLVM runtimes
support. This will automatically handle bootstrapping an up-to-date ``clang``
compiler and using it to build the C library. The following CMake invocation
will instruct it to build the ``libc`` runtime targeting both AMD and NVIDIA
GPUs. The ``LIBC_GPU_BUILD`` option can also be enabled to add the relevant
arguments automatically.

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build
  $> cd build
  $> cmake ../llvm -G Ninja                                                 \
     -DLLVM_ENABLE_PROJECTS="clang;lld"                                     \
     -DLLVM_ENABLE_RUNTIMES="openmp"                                        \
     -DCMAKE_BUILD_TYPE=<Debug|Release>   \ # Select build type
     -DCMAKE_INSTALL_PREFIX=<PATH>        \ # Where the libraries will live
     -DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES=libc               \
     -DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES=libc                 \
     -DLLVM_RUNTIME_TARGETS="default;amdgcn-amd-amdhsa;nvptx64-nvidia-cuda"
  $> ninja install

We need ``clang`` to build the GPU C library and ``lld`` to link AMDGPU
executables, so we enable them in ``LLVM_ENABLE_PROJECTS``. We add ``openmp`` to
``LLVM_ENABLED_RUNTIMES`` so it is built for the default target and provides
OpenMP support. We then set ``RUNTIMES_<triple>_LLVM_ENABLE_RUNTIMES`` to enable
``libc`` for the GPU targets. The ``LLVM_RUNTIME_TARGETS`` sets the enabled
targets to build, in this case we want the default target and the GPU targets.
Note that if ``libc`` were included in ``LLVM_ENABLE_RUNTIMES`` it would build
targeting the default host environment as well.

Runtimes cross build
--------------------

For users wanting more direct control over the build process, the build steps
can be done manually instead. This build closely follows the instructions in the
:ref:`main documentation<runtimes_cross_build>` but is specialized for the GPU
build. We follow the same steps to first build the libc tools and a suitable
compiler. These tools must all be up-to-date with the libc source.

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build-libc-tools # A different build directory for the build tools
  $> cd build-libc-tools
  $> HOST_C_COMPILER=<C compiler for the host> # For example "clang"
  $> HOST_CXX_COMPILER=<C++ compiler for the host> # For example "clang++"
  $> cmake ../llvm                            \
     -G Ninja                                 \
     -DLLVM_ENABLE_PROJECTS="clang;libc"      \
     -DCMAKE_C_COMPILER=$HOST_C_COMPILER      \
     -DCMAKE_CXX_COMPILER=$HOST_CXX_COMPILER  \
     -DLLVM_LIBC_FULL_BUILD=ON                \
     -DLIBC_HDRGEN_ONLY=ON    \ # Only build the 'libc-hdrgen' tool
     -DCMAKE_BUILD_TYPE=Release # Release suggested to make "clang" fast
  $> ninja # Build the 'clang' compiler
  $> ninja libc-hdrgen # Build the 'libc-hdrgen' tool

Once this has finished the build directory should contain the ``clang`` compiler
and the ``libc-hdrgen`` executable. We will use the ``clang`` compiler to build
the GPU code and the ``libc-hdrgen`` tool to create the necessary headers. We
use these tools to bootstrap the build out of the runtimes directory targeting a
GPU architecture.

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build # A different build directory for the build tools
  $> cd build
  $> TARGET_TRIPLE=<amdgcn-amd-amdhsa or nvptx64-nvidia-cuda>
  $> TARGET_C_COMPILER=</path/to/clang>
  $> TARGET_CXX_COMPILER=</path/to/clang++>
  $> HDRGEN=</path/to/libc-hdrgen>
  $> cmake ../runtimes \ # Point to the runtimes build
     -G Ninja                                  \
     -DLLVM_ENABLE_RUNTIMES=libc               \
     -DCMAKE_C_COMPILER=$TARGET_C_COMPILER     \
     -DCMAKE_CXX_COMPILER=$TARGET_CXX_COMPILER \
     -DLLVM_LIBC_FULL_BUILD=ON                 \
     -DLLVM_RUNTIMES_TARGET=$TARGET_TRIPLE     \
     -DLIBC_HDRGEN_EXE=$HDRGEN                 \
     -DCMAKE_BUILD_TYPE=Release
  $> ninja install

The above steps will result in a build targeting one of the supported GPU
architectures. Building for multiple targets requires separate CMake
invocations.

Standalone cross build
----------------------

The GPU build can also be targeted directly as long as the compiler used is a
supported ``clang`` compiler. This method is generally not recommended as it can
only target a single GPU architecture.

.. code-block:: sh

  $> cd llvm-project  # The llvm-project checkout
  $> mkdir build # A different build directory for the build tools
  $> cd build
  $> CLANG_C_COMPILER=</path/to/clang> # Must be a trunk build
  $> CLANG_CXX_COMPILER=</path/to/clang++> # Must be a trunk build
  $> TARGET_TRIPLE=<amdgcn-amd-amdhsa or nvptx64-nvidia-cuda>
  $> cmake ../llvm \ # Point to the llvm directory
     -G Ninja                                 \
     -DLLVM_ENABLE_PROJECTS=libc              \
     -DCMAKE_C_COMPILER=$CLANG_C_COMPILER     \
     -DCMAKE_CXX_COMPILER=$CLANG_CXX_COMPILER \
     -DLLVM_LIBC_FULL_BUILD=ON                \
     -DLIBC_TARGET_TRIPLE=$TARGET_TRIPLE      \
     -DCMAKE_BUILD_TYPE=Release
  $> ninja install

This will build and install the GPU C library along with all the other LLVM
libraries.

Build overview
==============

Once installed, the GPU build will create several files used for different
targets. This section will briefly describe their purpose.

**lib/<host-triple>/libcgpu-amdgpu.a or lib/libcgpu-amdgpu.a**
  A static library containing fat binaries supporting AMD GPUs. These are built
  using the support described in the `clang documentation
  <https://clang.llvm.org/docs/OffloadingDesign.html>`_. These are intended to
  be static libraries included natively for offloading languages like CUDA, HIP,
  or OpenMP. This implements the standard C library.

**lib/<host-triple>/libmgpu-amdgpu.a or lib/libmgpu-amdgpu.a**
  A static library containing fat binaries that implements the standard math
  library for AMD GPUs.

**lib/<host-triple>/libcgpu-nvptx.a or lib/libcgpu-nvptx.a**
  A static library containing fat binaries that implement the standard C library
  for NVIDIA GPUs.

**lib/<host-triple>/libmgpu-nvptx.a or lib/libmgpu-nvptx.a**
  A static library containing fat binaries that implement the standard math
  library for NVIDIA GPUs.

**include/<target-triple>**
  The include directory where all of the generated headers for the target will
  go. These definitions are strictly for the GPU when being targeted directly.

**lib/clang/<llvm-major-version>/include/llvm-libc-wrappers/llvm-libc-decls**
  These are wrapper headers created for offloading languages like CUDA, HIP, or
  OpenMP. They contain functions supported in the GPU libc along with attributes
  and metadata that declare them on the target device and make them compatible
  with the host headers.

**lib/<target-triple>/libc.a**
  The main C library static archive containing LLVM-IR targeting the given GPU.
  It can be linked directly or inspected depending on the target support.

**lib/<target-triple>/libm.a**
  The C library static archive providing implementations of the standard math
  functions.

**lib/<target-triple>/libc.bc**
  An alternate form of the library provided as a single LLVM-IR bitcode blob.
  This can be used similarly to NVIDIA's or AMD's device libraries.

**lib/<target-triple>/libm.bc**
  An alternate form of the library provided as a single LLVM-IR bitcode blob
  containing the standard math functions.

**lib/<target-triple>/crt1.o**
  An LLVM-IR file containing startup code to call the ``main`` function on the
  GPU. This is used similarly to the standard C library startup object.

**bin/amdhsa-loader**
  A binary utility used to launch executables compiled targeting the AMD GPU.
  This will be included if the build system found the ``hsa-runtime64`` library
  either in ``/opt/rocm`` or the current CMake installation directory. This is
  required to build the GPU tests .See the :ref:`libc GPU usage<libc_gpu_usage>`
  for more information.

**bin/nvptx-loader**
  A binary utility used to launch executables compiled targeting the NVIDIA GPU.
  This will be included if the build system found the CUDA driver API. This is
  required for building tests.

**include/llvm-libc-rpc-server.h**
  A header file containing definitions that can be used to interface with the
  :ref:`RPC server<libc_gpu_rpc>`.

**lib/libllvmlibc_rpc_server.a**
  The static library containing the implementation of the RPC server. This can
  be used to enable host services for anyone looking to interface with the
  :ref:`RPC client<libc_gpu_rpc>`.

.. _gpu_cmake_options:

CMake options
=============

This section briefly lists a few of the CMake variables that specifically
control the GPU build of the C library. These options can be passed individually
to each target using ``-DRUNTIMES_<target>_<variable>=<value>`` when using a
standard runtime build.

**LLVM_LIBC_FULL_BUILD**:BOOL
  This flag controls whether or not the libc build will generate its own
  headers. This must always be on when targeting the GPU.

**LIBC_GPU_BUILD**:BOOL
  Shorthand for enabling GPU support. Equivalent to enabling support for both
  AMDGPU and NVPTX builds for ``libc``.

**LIBC_GPU_TEST_ARCHITECTURE**:STRING
  Sets the architecture used to build the GPU tests for, such as ``gfx90a`` or
  ``sm_80`` for AMD and NVIDIA GPUs respectively. The default behavior is to
  detect the system's GPU architecture using the ``native`` option. If this
  option is not set and a GPU was not detected the tests will not be built.

**LIBC_GPU_TEST_JOBS**:STRING
  Sets the number of threads used to run GPU tests. The GPU test suite will
  commonly run out of resources if this is not constrained so it is recommended
  to keep it low. The default value is a single thread.

**LIBC_GPU_LOADER_EXECUTABLE**:STRING
  Overrides the default loader used for running GPU tests. If this is not
  provided the standard one will be built.
