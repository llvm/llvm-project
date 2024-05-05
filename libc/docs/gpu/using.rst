.. _libc_gpu_usage:

===================
Using libc for GPUs
===================

.. contents:: Table of Contents
  :depth: 4
  :local:

Using the GPU C library
=======================

Once you have finished :ref:`building<libc_gpu_building>` the GPU C library it
can be used to run libc or libm functions directly on the GPU. Currently, not
all C standard functions are supported on the GPU. Consult the :ref:`list of
supported functions<libc_gpu_support>` for a comprehensive list.

The GPU C library supports two main usage modes. The first is as a supplementary
library for offloading languages such as OpenMP, CUDA, or HIP. These aim to
provide standard system utilities similarly to existing vendor libraries. The
second method treats the GPU as a hosted target by compiling C or C++ for it
directly. This is more similar to targeting OpenCL and is primarily used for
exported functions on the GPU and testing.

Offloading usage
----------------

Offloading languages like CUDA, HIP, or OpenMP work by compiling a single source
file for both the host target and a list of offloading devices. In order to
support standard compilation flows, the ``clang`` driver uses fat binaries,
described in the `clang documentation
<https://clang.llvm.org/docs/OffloadingDesign.html>`_. This linking mode is used
by the OpenMP toolchain, but is currently opt-in for the CUDA and HIP toolchains
through the ``--offload-new-driver``` and ``-fgpu-rdc`` flags.

The installation should contain a static library called ``libcgpu-amdgpu.a`` or
``libcgpu-nvptx.a`` depending on which GPU architectures your build targeted.
These contain fat binaries compatible with the offloading toolchain such that
they can be used directly.

.. code-block:: sh

  $> clang opnemp.c -fopenmp --offload-arch=gfx90a -lcgpu-amdgpu
  $> clang cuda.cu --offload-arch=sm_80 --offload-new-driver -fgpu-rdc -lcgpu-nvptx
  $> clang hip.hip --offload-arch=gfx940 --offload-new-driver -fgpu-rdc -lcgpu-amdgpu

This will automatically link in the needed function definitions if they were
required by the user's application. Normally using the ``-fgpu-rdc`` option
results in sub-par performance due to ABA linking. However, the offloading
toolchain supports the ``--foffload-lto`` option to support LTO on the target
device.

Offloading languages require that functions present on the device be declared as
such. This is done with the ``__device__`` keyword in CUDA and HIP or the
``declare target`` pragma in OpenMP. This requires that the LLVM C library
exposes its implemented functions to the compiler when it is used to build. We
support this by providing wrapper headers in the compiler's resource directory.
These are located in ``<clang-resource-dir>/include/llvm-libc-wrappers`` in your
installation.

The support for HIP and CUDA is more experimental, requiring manual intervention
to link and use the facilities. An example of this is shown in the :ref:`CUDA
server example<libc_gpu_cuda_server>`. The OpenMP Offloading toolchain is
completely integrated with the LLVM C library however. It will automatically
handle including the necessary libraries, define device-side interfaces, and run
the RPC server.

OpenMP Offloading example
^^^^^^^^^^^^^^^^^^^^^^^^^

This section provides a simple example of compiling an OpenMP program with the
GPU C library.

.. code-block:: c++

  #include <stdio.h>

  int main() {
    FILE *file = stderr;
  #pragma omp target teams num_teams(2) thread_limit(2)
  #pragma omp parallel num_threads(2)
    { fputs("Hello from OpenMP!\n", file); }
  }

This can simply be compiled like any other OpenMP application to print from two
threads and two blocks.

.. code-block:: sh

  $> clang openmp.c -fopenmp --offload-arch=gfx90a
  $> ./a.out
  Hello from OpenMP!
  Hello from OpenMP!
  Hello from OpenMP!
  Hello from OpenMP!

Including the wrapper headers, linking the C library, and running the :ref:`RPC
server<libc_gpu_rpc>` are all handled automatically by the compiler and runtime.

Binary format
^^^^^^^^^^^^^

The ``libcgpu.a`` static archive is a fat-binary containing LLVM-IR for each
supported target device. The supported architectures can be seen using LLVM's
``llvm-objdump`` with the ``--offloading`` flag:

.. code-block:: sh

  $> llvm-objdump --offloading libcgpu-amdgpu.a
  libcgpu-amdgpu.a(strcmp.cpp.o):    file format elf64-x86-64

  OFFLOADING IMAGE [0]:
  kind            llvm ir
  arch            generic
  triple          amdgcn-amd-amdhsa
  producer        none
  ...

Because the device code is stored inside a fat binary, it can be difficult to
inspect the resulting code. This can be done using the following utilities:

.. code-block:: sh

  $> llvm-ar x libcgpu.a strcmp.cpp.o
  $> clang-offload-packager strcmp.cpp.o --image=arch=generic,file=strcmp.bc
  $> opt -S out.bc
  ...

Please note that this fat binary format is provided for compatibility with
existing offloading toolchains. The implementation in ``libc`` does not depend
on any existing offloading languages and is completely freestanding.

Direct compilation
------------------

Instead of using standard offloading languages, we can also target the CPU
directly using C and C++ to create a GPU executable similarly to OpenCL. This is
done by targeting the GPU architecture using `clang's cross compilation
support <https://clang.llvm.org/docs/CrossCompilation.html>`_. This is the
method that the GPU C library uses both to build the library and to run tests.

This allows us to easily define GPU specific libraries and programs that fit
well into existing tools. In order to target the GPU effectively we rely heavily
on the compiler's intrinsic and built-in functions. For example, the following
function gets the thread identifier in the 'x' dimension on both GPUs supported
GPUs.

.. code-block:: c++

  uint32_t get_thread_id_x() {
  #if defined(__AMDGPU__)
    return __builtin_amdgcn_workitem_id_x();
  #elif defined(__NVPTX__)
    return __nvvm_read_ptx_sreg_tid_x();
  #else
  #error "Unsupported platform"
  #endif
  }

We can then compile this for both NVPTX and AMDGPU into LLVM-IR using the
following commands. This will yield valid LLVM-IR for the given target just like
if we were using CUDA, OpenCL, or OpenMP.

.. code-block:: sh

  $> clang id.c --target=amdgcn-amd-amdhsa -mcpu=native -nogpulib -flto -c
  $> clang id.c --target=nvptx64-nvidia-cuda -march=native -nogpulib -flto -c

We can also use this support to treat the GPU as a hosted environment by
providing a C library and startup object just like a standard C library running
on the host machine. Then, in order to execute these programs, we provide a
loader utility to launch the executable on the GPU similar to a cross-compiling
emulator. This is how we run :ref:`unit tests <libc_gpu_testing>` targeting the
GPU. This is clearly not the most efficient way to use a GPU, but it provides a
simple method to test execution on a GPU for debugging or development.

Building for AMDGPU targets
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The AMDGPU target supports several features natively by virtue of using ``lld``
as its linker. The installation will include the ``include/amdgcn-amd-amdhsa``
and ``lib/amdgcn-amd-amdha`` directories that contain the necessary code to use
the library. We can directly link against ``libc.a`` and use LTO to generate the
final executable.

.. code-block:: c++

  #include <stdio.h>

  int main() { fputs("Hello from AMDGPU!\n", stdout); }

This program can then be compiled using the ``clang`` compiler. Note that
``-flto`` and ``-mcpu=`` should be defined. This is because the GPU
sub-architectures do not have strict backwards compatibility. Use ``-mcpu=help``
for accepted arguments or ``-mcpu=native`` to target the system's installed GPUs
if present. Additionally, the AMDGPU target always uses ``-flto`` because we
currently do not fully support ELF linking in ``lld``. Once built, we use the
``amdhsa-loader`` utility to launch execution on the GPU. This will be built if
the ``hsa_runtime64`` library was found during build time.

.. code-block:: sh

  $> clang hello.c --target=amdgcn-amd-amdhsa -mcpu=native -flto -lc <install>/lib/amdgcn-amd-amdhsa/crt1.o
  $> amdhsa-loader --threads 2 --blocks 2 a.out
  Hello from AMDGPU!
  Hello from AMDGPU!
  Hello from AMDGPU!
  Hello from AMDGPU!

This will include the ``stdio.h`` header, which is found in the
``include/amdgcn-amd-amdhsa`` directory. We define out ``main`` function like a
standard application. The startup utility in ``lib/amdgcn-amd-amdhsa/crt1.o``
will handle the necessary steps to execute the ``main`` function along with
global initializers and command line arguments. Finally, we link in the
``libc.a`` library stored in ``lib/amdgcn-amd-amdhsa`` to define the standard C
functions.

The search paths for the include directories and libraries are automatically
handled by the compiler. We use this support internally to run unit tests on the
GPU directly. See :ref:`libc_gpu_testing` for more information. The installation
also provides ``libc.bc`` which is a single LLVM-IR bitcode blob that can be
used instead of the static library.

Building for NVPTX targets
^^^^^^^^^^^^^^^^^^^^^^^^^^

The infrastructure is the same as the AMDGPU example. However, the NVPTX binary
utilities are very limited and must be targeted directly. There is no linker
support for static libraries so we need to link in the ``libc.bc`` bitcode and
inform the compiler driver of the file's contents.

.. code-block:: c++

  #include <stdio.h>

  int main(int argc, char **argv, char **envp) {
    fputs("Hello from NVPTX!\n", stdout);
  }

Additionally, the NVPTX ABI requires that every function signature matches. This
requires us to pass the full prototype from ``main``. The installation will
contain the ``nvptx-loader`` utility if the CUDA driver was found during
compilation.

.. code-block:: sh

  $> clang hello.c --target=nvptx64-nvidia-cuda -march=native \
       -x ir <install>/lib/nvptx64-nvidia-cuda/libc.bc \
       -x ir <install>/lib/nvptx64-nvidia-cuda/crt1.o
  $> nvptx-loader --threads 2 --blocks 2 a.out
  Hello from NVPTX!
  Hello from NVPTX!
  Hello from NVPTX!
  Hello from NVPTX!
