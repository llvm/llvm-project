.. raw:: html

  <style type="text/css">
    .none { background-color: #FFCCCC }
    .part { background-color: #FFFF99 }
    .good { background-color: #CCFF99 }
  </style>

.. role:: none
.. role:: part
.. role:: good

.. contents::
   :local:

=============
HIP Support
=============

HIP (Heterogeneous-Compute Interface for Portability) `<https://github.com/ROCm-Developer-Tools/HIP>`_ is
a C++ Runtime API and Kernel Language. It enables developers to create portable applications for
offloading computation to different hardware platforms from a single source code.

AMD GPU Support
===============

Clang provides HIP support on AMD GPUs via the ROCm platform `<https://rocm.docs.amd.com/en/latest/#>`_.
The ROCm runtime forms the base for HIP host APIs, while HIP device APIs are realized through HIP header
files and the ROCm device library. The Clang driver uses the HIPAMD toolchain to compile HIP device code
to AMDGPU ISA via the AMDGPU backend. The compiled code is then bundled and embedded in the host executables.

Intel GPU Support
=================

Clang provides partial HIP support on Intel GPUs using the CHIP-Star project `<https://github.com/CHIP-SPV/chipStar>`_.
CHIP-Star implements the HIP runtime over oneAPI Level Zero or OpenCL runtime. The Clang driver uses the HIPSPV
toolchain to compile HIP device code into LLVM IR, which is subsequently translated to SPIR-V via the SPIR-V
backend or the out-of-tree LLVM-SPIRV translator. The SPIR-V is then bundled and embedded into the host executables.

.. note::
   While Clang does not directly provide HIP support for NVIDIA GPUs and CPUs, these platforms are supported via other means:

   - NVIDIA GPUs: HIP support is offered through the HIP project `<https://github.com/ROCm-Developer-Tools/HIP>`_, which provides a header-only library for translating HIP runtime APIs into CUDA runtime APIs. The code is subsequently compiled using NVIDIA's `nvcc`.

   - CPUs: HIP support is available through the HIP-CPU runtime library `<https://github.com/ROCm-Developer-Tools/HIP-CPU>`_. This header-only library enables CPUs to execute unmodified HIP code.


Example Usage
=============

To compile a HIP program, use the following command:

.. code-block:: shell

   clang++ -c --offload-arch=gfx906 -xhip sample.cpp -o sample.o

The ``-xhip`` option indicates that the source is a HIP program. If the file has a ``.hip`` extension,
Clang will automatically recognize it as a HIP program:

.. code-block:: shell

   clang++ -c --offload-arch=gfx906 sample.hip -o sample.o

To link a HIP program, use this command:

.. code-block:: shell

   clang++ --hip-link --offload-arch=gfx906 sample.o -o sample

In the above command, the ``--hip-link`` flag instructs Clang to link the HIP runtime library. However,
the use of this flag is unnecessary if a HIP input file is already present in your program.

For convenience, Clang also supports compiling and linking in a single step:

.. code-block:: shell

   clang++ --offload-arch=gfx906 -xhip sample.cpp -o sample

In the above commands, ``gfx906`` is the GPU architecture that the code is being compiled for. The supported GPU
architectures can be found in the `AMDGPU Processor Table <https://llvm.org/docs/AMDGPUUsage.html#processors>`_.
Alternatively, you can use the ``amdgpu-arch`` tool that comes with Clang to list the GPU architecture on your system:

.. code-block:: shell

   amdgpu-arch

You can use ``--offload-arch=native`` to automatically detect the GPU architectures on your system:

.. code-block:: shell

   clang++ --offload-arch=native -xhip sample.cpp -o sample


Path Setting for Dependencies
=============================

Compiling a HIP program depends on the HIP runtime and device library. The paths to the HIP runtime and device libraries
can be specified either using compiler options or environment variables. The paths can also be set through the ROCm path
if they follow the ROCm installation directory structure.

Order of Precedence for HIP Path
--------------------------------

1. ``--hip-path`` compiler option
2. ``HIP_PATH`` environment variable *(use with caution)*
3. ``--rocm-path`` compiler option
4. ``ROCM_PATH`` environment variable *(use with caution)*
5. Default automatic detection (relative to Clang or at the default ROCm installation location)

Order of Precedence for Device Library Path
-------------------------------------------

1. ``--hip-device-lib-path`` compiler option
2. ``HIP_DEVICE_LIB_PATH`` environment variable *(use with caution)*
3. ``--rocm-path`` compiler option
4. ``ROCM_PATH`` environment variable *(use with caution)*
5. Default automatic detection (relative to Clang or at the default ROCm installation location)

.. list-table::
   :header-rows: 1

   * - Compiler Option
     - Environment Variable
     - Description
     - Default Value
   * - ``--rocm-path=<path>``
     - ``ROCM_PATH``
     - Specifies the ROCm installation path.
     - Automatic detection
   * - ``--hip-path=<path>``
     - ``HIP_PATH``
     - Specifies the HIP runtime installation path.
     - Determined by ROCm directory structure
   * - ``--hip-device-lib-path=<path>``
     - ``HIP_DEVICE_LIB_PATH``
     - Specifies the HIP device library installation path.
     - Determined by ROCm directory structure

.. note::

   We recommend using the compiler options as the primary method for specifying these paths. While the environment variables ``ROCM_PATH``, ``HIP_PATH``, and ``HIP_DEVICE_LIB_PATH`` are supported, their use can lead to implicit dependencies that might cause issues in the long run. Use them with caution.


Predefined Macros
=================

.. list-table::
   :header-rows: 1

   * - Macro
     - Description
   * - ``__CLANG_RDC__``
     - Defined when Clang is compiling code in Relocatable Device Code (RDC) mode. RDC, enabled with the ``-fgpu-rdc`` compiler option, is necessary for linking device codes across translation units.
   * - ``__HIP__``
     - Defined when compiling with HIP language support, indicating that the code targets the HIP environment.
   * - ``__HIPCC__``
     - Alias to ``__HIP__``.
   * - ``__HIP_DEVICE_COMPILE__``
     - Defined during device code compilation in Clang's separate compilation process for the host and each offloading GPU architecture.
   * - ``__HIP_MEMORY_SCOPE_SINGLETHREAD``
     - Represents single-thread memory scope in HIP (value is 1).
   * - ``__HIP_MEMORY_SCOPE_WAVEFRONT``
     - Represents wavefront memory scope in HIP (value is 2).
   * - ``__HIP_MEMORY_SCOPE_WORKGROUP``
     - Represents workgroup memory scope in HIP (value is 3).
   * - ``__HIP_MEMORY_SCOPE_AGENT``
     - Represents agent memory scope in HIP (value is 4).
   * - ``__HIP_MEMORY_SCOPE_SYSTEM``
     - Represents system-wide memory scope in HIP (value is 5).
   * - ``__HIP_NO_IMAGE_SUPPORT__``
     - Defined with a value of 1 when the target device lacks support for HIP image functions.
   * - ``__HIP_NO_IMAGE_SUPPORT``
     - Alias to ``__HIP_NO_IMAGE_SUPPORT__``. Deprecated.
   * - ``__HIP_API_PER_THREAD_DEFAULT_STREAM__``
     - Defined when the GPU default stream is set to per-thread mode.
   * - ``HIP_API_PER_THREAD_DEFAULT_STREAM``
     - Alias to ``__HIP_API_PER_THREAD_DEFAULT_STREAM__``. Deprecated.

Support for Deduction Guides
============================

Explicit Deduction Guides
-------------------------

Explicit deduction guides in HIP can be annotated with either the
``__host__`` or ``__device__`` attributes. If no attribute is provided,
it defaults to ``__host__``.

.. code-block:: cpp

   template <typename T>
   class MyArray {
       //...
   };

   template <typename T>
   MyArray(T)->MyArray<T>;

   __device__ MyArray(float)->MyArray<int>;

   // Uses of the deduction guides
   MyArray arr1 = 10;      // Uses the default host guide
   __device__ void foo() {
       MyArray arr2 = 3.14f; // Uses the device guide
   }

Implicit Deduction Guides
-------------------------
Implicit deduction guides derived from constructors inherit the same host or
device attributes as the originating constructor.

.. code-block:: cpp

   template <typename T>
   class MyVector {
   public:
       __device__ MyVector(T) { /* ... */ }
       //...
   };

   // The implicit deduction guide for MyVector will be `__device__` due to the device constructor

   __device__ void foo() {
       MyVector vec(42);  // Uses the implicit device guide derived from the constructor
   }

Availability Checks
--------------------
When a deduction guide (either explicit or implicit) is used, HIP checks its
availability based on its host/device attributes and the context in a similar
way as checking a function. Utilizing a deduction guide in an incompatible context
results in a compile-time error.

