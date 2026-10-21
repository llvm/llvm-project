.. _libc_gpu_building:

======================
Building libs for GPUs
======================

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

Bootstrap Build
---------------

The recommended way to build the GPU libc is using a **Bootstrap Build** (see :ref:`build_concepts` for an overview of build modes). This approach first builds the compiler (``clang``) and tools using the host compiler, and then automatically uses that newly-built compiler to build the C library for the GPU targets in a single CMake invocation. This is ideal for building for multiple GPU vendors at once.

Set the environment variables for the build:

.. code-block:: sh

  export BUILD_DIR=build
  export INSTALL_PREFIX=install
  export BUILD_TYPE=Release

Configure the project:

.. code-block:: sh

  cmake -G Ninja -S llvm -B $BUILD_DIR \
     -DLLVM_ENABLE_PROJECTS="clang;lld" \
     -DLLVM_ENABLE_RUNTIMES="openmp;offload" \
     -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
     -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
     -DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES=libc \
     -DRUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES=libc \
     -DLLVM_RUNTIME_TARGETS="default;amdgcn-amd-amdhsa;nvptx64-nvidia-cuda"

Build and install the libraries:

.. code-block:: sh

  ninja -C $BUILD_DIR install

We need ``clang`` to build the GPU C library and ``lld`` to link AMDGPU
executables, so we enable them in ``LLVM_ENABLE_PROJECTS``. We add ``openmp`` to
``LLVM_ENABLED_RUNTIMES`` so it is built for the default target and provides
OpenMP support. We then set ``RUNTIMES_<triple>_LLVM_ENABLE_RUNTIMES`` to enable
``libc`` for the GPU targets. The ``LLVM_RUNTIME_TARGETS`` sets the enabled
targets to build, in this case we want the default target and the GPU targets.
Note that if ``libc`` were included in ``LLVM_ENABLE_RUNTIMES`` it would build
targeting the default host environment as well. Alternatively, you can point
your build towards the ``libc/cmake/caches/gpu.cmake`` cache file with ``-C``.

Two-stage Cross-compiler Build
------------------------------

Alternatively, you can use a manual **Two-stage Cross-compiler Build** (see :ref:`build_concepts`). This separates the build into two distinct CMake invocations: first to build the host compiler and tools (Stage 1), and then to build the target library using those tools (Stage 2). This provides more direct control over the configuration and is useful if you only need to build for a single target or want to avoid the complexity of the combined bootstrap build.

First, build the host compiler. Set up the variables for the host build:

.. code-block:: sh

  export HOST_BUILD_DIR=build-libc-tools
  export HOST_C_COMPILER=clang
  export HOST_CXX_COMPILER=clang++

Configure the host build:

.. code-block:: sh

  cmake -G Ninja -S llvm -B $HOST_BUILD_DIR \
     -DLLVM_ENABLE_PROJECTS="clang" \
     -DCMAKE_C_COMPILER=$HOST_C_COMPILER \
     -DCMAKE_CXX_COMPILER=$HOST_CXX_COMPILER \
     -DLLVM_LIBC_FULL_BUILD=ON \
     -DCMAKE_BUILD_TYPE=Release

Build the host tools:

.. code-block:: sh

  ninja -C $HOST_BUILD_DIR

Once this has finished, use the newly built compiler to build the C library for the GPU. Select your target architecture (``amdgcn-amd-amdhsa`` or ``nvptx64-nvidia-cuda``).

Set up the variables for the target build:

.. code-block:: sh

  export TARGET_TRIPLE=amdgcn-amd-amdhsa # or nvptx64-nvidia-cuda
  export TARGET_BUILD_DIR=build
  export TARGET_C_COMPILER=build-libc-tools/bin/clang
  export TARGET_CXX_COMPILER=build-libc-tools/bin/clang++

Configure the target build:

.. code-block:: sh

  cmake -G Ninja -S runtimes -B $TARGET_BUILD_DIR \
     -DLLVM_ENABLE_RUNTIMES=libc \
     -DCMAKE_C_COMPILER=$TARGET_C_COMPILER \
     -DCMAKE_CXX_COMPILER=$TARGET_CXX_COMPILER   \
     -DLLVM_LIBC_FULL_BUILD=ON                   \
     -DLLVM_DEFAULT_TARGET_TRIPLE=$TARGET_TRIPLE \
     -DCMAKE_BUILD_TYPE=Release

Build and install the target library:

.. code-block:: sh

  ninja -C $TARGET_BUILD_DIR install

The above steps will result in a build targeting one of the supported GPU
architectures. Building for multiple targets requires separate CMake
invocations.

Build overview
==============

Once installed, the GPU build will create several files used for different
targets. This section will briefly describe their purpose.

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

**CMAKE_CROSSCOMPILING_EMULATOR**:STRING
  Overrides the default loader used for running GPU tests. This is set
  automatically to ``llvm-gpu-loader`` for GPU runtime targets when building
  via the runtimes build.
