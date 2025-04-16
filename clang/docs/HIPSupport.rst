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
to AMDGPU ISA via the AMDGPU backend, or SPIR-V via the workflow outlined below.
The compiled code is then bundled and embedded in the host executables.

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

Note that some architecture specific AMDGPU macros will have default values when
used from the HIP host compilation. Other :doc:`AMDGPU macros <AMDGPUSupport>`
like ``__AMDGCN_WAVEFRONT_SIZE__`` (deprecated) will default to 64 for example.

Compilation Modes
=================

Each HIP source file contains intertwined device and host code. Depending on the chosen compilation mode by the compiler options ``-fno-gpu-rdc`` and ``-fgpu-rdc``, these portions of code are compiled differently.

Device Code Compilation
-----------------------

**``-fno-gpu-rdc`` Mode (default)**:

- Compiles to a self-contained, fully linked offloading device binary for each offloading device architecture.
- Device code within a Translation Unit (TU) cannot call functions located in another TU.

**``-fgpu-rdc`` Mode**:

- Compiles to a bitcode for each GPU architecture.
- For each offloading device architecture, the bitcode from different TUs are linked together to create a single offloading device binary.
- Device code in one TU can call functions located in another TU.

Host Code Compilation
---------------------

**Both Modes**:

- Compiles to a relocatable object for each TU.
- These relocatable objects are then linked together.
- Host code within a TU can call host functions and launch kernels from another TU.

Syntax Difference with CUDA
===========================

Clang's front end, used for both CUDA and HIP programming models, shares the same parsing and semantic analysis mechanisms. This includes the resolution of overloads concerning device and host functions. While there exists a comprehensive documentation on the syntax differences between Clang and NVCC for CUDA at `Dialect Differences Between Clang and NVCC <https://llvm.org/docs/CompileCudaWithLLVM.html#dialect-differences-between-clang-and-nvcc>`_, it is important to note that these differences also apply to HIP code compilation.

Predefined Macros for Differentiation
-------------------------------------

To facilitate differentiation between HIP and CUDA code, as well as between device and host compilations within HIP, Clang defines specific macros:

- ``__HIP__`` : This macro is defined only when compiling HIP code. It can be used to conditionally compile code specific to HIP, enabling developers to write portable code that can be compiled for both CUDA and HIP.

- ``__HIP_DEVICE_COMPILE__`` : Defined exclusively during HIP device compilation, this macro allows for conditional compilation of device-specific code. It provides a mechanism to segregate device and host code, ensuring that each can be optimized for their respective execution environments.

Function Pointers Support
=========================

Function pointers' support varies with the usage mode in Clang with HIP. The following table provides an overview of the support status across different use-cases and modes.

.. list-table:: Function Pointers Support Overview
   :widths: 25 25 25
   :header-rows: 1

   * - Use Case
     - ``-fno-gpu-rdc`` Mode (default)
     - ``-fgpu-rdc`` Mode
   * - Defined and used in the same TU
     - Supported
     - Supported
   * - Defined in one TU and used in another TU
     - Not Supported
     - Supported

In the ``-fno-gpu-rdc`` mode, the compiler calculates the resource usage of kernels based only on functions present within the same TU. This mode does not support the use of function pointers defined in a different TU due to the possibility of incorrect resource usage calculations, leading to undefined behavior.

On the other hand, the ``-fgpu-rdc`` mode allows the definition and use of function pointers across different TUs, as resource usage calculations can accommodate functions from disparate TUs.

Virtual Function Support
========================

In Clang with HIP, support for calling virtual functions of an object in device or host code is contingent on where the object is constructed.

- **Constructed in Device Code**: Virtual functions of an object can be called in device code on a specific offloading device if the object is constructed in device code on an offloading device with the same architecture.
- **Constructed in Host Code**: Virtual functions of an object can be called in host code if the object is constructed in host code.

In other scenarios, calling virtual functions is not allowed.

Explanation
-----------

An object constructed on the device side contains a pointer to the virtual function table on the device side, which is not accessible in host code, and vice versa. Thus, trying to invoke virtual functions from a context different from where the object was constructed will be disallowed because the appropriate virtual table cannot be accessed. The virtual function tables for offloading devices with different architecures are different, therefore trying to invoke virtual functions from an offloading device with a different architecture than where the object is constructed is also disallowed.

Example Usage
-------------

.. code-block:: c++

   class Base {
   public:
      __device__ virtual void virtualFunction() {
         // Base virtual function implementation
      }
   };

   class Derived : public Base {
   public:
      __device__ void virtualFunction() override {
         // Derived virtual function implementation
      }
   };

   __global__ void kernel() {
      Derived obj;
      Base* basePtr = &obj;
      basePtr->virtualFunction(); // Allowed since obj is constructed in device code
   }

C++ Standard Parallelism Offload Support: Compiler And Runtime
==============================================================

Introduction
============

This section describes the implementation of support for offloading the
execution of standard C++ algorithms to accelerators that can be targeted via
HIP. Furthermore, it enumerates restrictions on user defined code, as well as
the interactions with runtimes.

Algorithm Offload: What, Why, Where
===================================

C++17 introduced overloads
`for most algorithms in the standard library <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0024r2.html>`_
which allow the user to specify a desired
`execution policy <https://en.cppreference.com/w/cpp/algorithm#Execution_policies>`_.
The `parallel_unsequenced_policy <https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t>`_
maps relatively well to the execution model of AMD GPUs. This, coupled with the
the availability and maturity of GPU accelerated algorithm libraries that
implement most / all corresponding algorithms in the standard library
(e.g. `rocThrust <https://github.com/ROCmSoftwarePlatform/rocThrust>`__), makes
it feasible to provide seamless accelerator offload for supported algorithms,
when an accelerated version exists. Thus, it becomes possible to easily access
the computational resources of an AMD accelerator, via a well specified,
familiar, algorithmic interface, without having to delve into low-level hardware
specific details. Putting it all together:

- **What**: standard library algorithms, when invoked with the
  ``parallel_unsequenced_policy``
- **Why**: democratise AMDGPU accelerator programming, without loss of user
  familiarity
- **Where**: only AMDGPU accelerators targeted by Clang/LLVM via HIP

Small Example
=============

Given the following C++ code:

.. code-block:: C++

   bool has_the_answer(const std::vector<int>& v) {
     return std::find(std::execution::par_unseq, std::cbegin(v), std::cend(v), 42) != std::cend(v);
   }

if Clang is invoked with the ``--hipstdpar --offload-arch=foo`` flags, the call
to ``find`` will be offloaded to an accelerator that is part of the ``foo``
target family. If either ``foo`` or its runtime environment do not support
transparent on-demand paging (such as e.g. that provided in Linux via
`HMM <https://docs.kernel.org/mm/hmm.html>`_), it is necessary to also include
the ``--hipstdpar-interpose-alloc`` flag. If the accelerator specific algorithm
library ``foo`` uses doesn't have an implementation of a particular algorithm,
execution seamlessly falls back to the host CPU. It is legal to specify multiple
``--offload-arch``\s. All the flags we introduce, as well as a thorough view of
various restrictions an their implementations, will be provided below.

Implementation - General View
=============================

We built support for Algorithm Offload support atop the pre-existing HIP
infrastructure. More specifically, when one requests offload via ``--hipstdpar``,
compilation is switched to HIP compilation, as if ``-x hip`` was specified.
Similarly, linking is also switched to HIP linking, as if ``--hip-link`` was
specified. Note that these are implicit, and one should not assume that any
interop with HIP specific language constructs is available e.g. ``__device__``
annotations are neither necessary nor guaranteed to work.

Since there are no language restriction mechanisms in place, it is necessary to
relax HIP language specific semantic checks performed by the FE; they would
identify otherwise valid, offloadable code, as invalid HIP code. Given that we
know that the user intended only for certain algorithms to be offloaded, and
encoded this by specifying the ``parallel_unsequenced_policy``, we rely on a
pass over IR to clean up any and all code that was not "meant" for offload. If
requested, allocation interposition is also handled via a separate pass over IR.

To interface with the client HIP runtime, and to forward offloaded algorithm
invocations to the corresponding accelerator specific library implementation, an
implementation detail forwarding header is implicitly included by the driver,
when compiling with ``--hipstdpar``. In what follows, we will delve into each
component that contributes to implementing Algorithm Offload support.

Implementation - Driver
=======================

We augment the ``clang`` driver with the following flags:

- ``--hipstdpar`` enables algorithm offload, which depending on phase, has the
  following effects:

  - when compiling:

    - ``-x hip`` gets prepended to enable HIP support;
    - the ``ROCmToolchain`` component checks for the ``hipstdpar_lib.hpp``
      forwarding header,
      `rocThrust <https://rocm.docs.amd.com/projects/rocThrust/en/latest/>`_ and
      `rocPrim <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/>`_ in
      their canonical locations, which can be overriden via flags found below;
      if all are found, the forwarding header gets implicitly included,
      otherwise an error listing the missing component is generated;
    - the ``LangOpts.HIPStdPar`` member is set.

  - when linking:

    - ``--hip-link`` and ``-frtlib-add-rpath`` gets appended to enable HIP
      support.

- ``--hipstdpar-interpose-alloc`` enables the interposition of standard
  allocation / deallocation functions with accelerator aware equivalents; the
  ``LangOpts.HIPStdParInterposeAlloc`` member is set;
- ``--hipstdpar-path=`` specifies a non-canonical path for the forwarding
  header; it must point to the folder where the header is located and not to the
  header itself;
- ``--hipstdpar-thrust-path=`` specifies a non-canonical path for
  `rocThrust <https://rocm.docs.amd.com/projects/rocThrust/en/latest/>`_; it
  must point to the folder where the library is installed / built under a
  ``/thrust`` subfolder;
- ``--hipstdpar-prim-path=`` specifies a non-canonical path for
  `rocPrim <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/>`_; it must
  point to the folder where the library is installed / built under a
  ``/rocprim`` subfolder;

The `--offload-arch <https://llvm.org/docs/AMDGPUUsage.html#amdgpu-processors>`_
flag can be used to specify the accelerator for which offload code is to be
generated.

Implementation - Front-End
==========================

When ``LangOpts.HIPStdPar`` is set, we relax some of the HIP language specific
``Sema`` checks to account for the fact that we want to consume pure unannotated
C++ code:

1. ``__device__`` / ``__host__ __device__`` functions (which would originate in
   the accelerator specific algorithm library) are allowed to call implicitly
   ``__host__`` functions;
2. ``__global__`` functions (which would originate in the accelerator specific
   algorithm library) are allowed to call implicitly ``__host__`` functions;
3. resolving ``__builtin`` availability is deferred, because it is possible that
   a ``__builtin`` that is unavailable on the target accelerator is not
   reachable from any offloaded algorithm, and thus will be safely removed in
   the middle-end;
4. ASM parsing / checking is deferred, because it is possible that an ASM block
   that e.g. uses some constraints that are incompatible with the target
   accelerator is not reachable from any offloaded algorithm, and thus will be
   safely removed in the middle-end.

``CodeGen`` is similarly relaxed, with implicitly ``__host__`` functions being
emitted as well.

Implementation - Middle-End
===========================

We add two ``opt`` passes:

1. ``HipStdParAcceleratorCodeSelectionPass``

   - For all kernels in a ``Module``, compute reachability, where a function
     ``F`` is reachable from a kernel ``K`` if and only if there exists a direct
     call-chain rooted in ``F`` that includes ``K``;
   - Remove all functions that are not reachable from kernels;
   - This pass is only run when compiling for the accelerator.

The first pass assumes that the only code that the user intended to offload was
that which was directly or transitively invocable as part of an algorithm
execution. It also assumes that an accelerator aware algorithm implementation
would rely on accelerator specific special functions (kernels), and that these
effectively constitute the only roots for accelerator execution graphs. Both of
these assumptions are based on observing how widespread accelerators,
such as GPUs, work.

1. ``HipStdParAllocationInterpositionPass``

   - Iterate through all functions in a ``Module``, and replace standard
     allocation / deallocation functions with accelerator-aware equivalents,
     based on a pre-established table; the list of functions that can be
     interposed is available
     `here <https://github.com/ROCmSoftwarePlatform/roc-stdpar#allocation--deallocation-interposition-status>`__;
   - This is only run when compiling for the host.

The second pass is optional.

Implementation - Forwarding Header
==================================

The forwarding header implements two pieces of functionality:

1. It forwards algorithms to a target accelerator, which is done by relying on
   C++ language rules around overloading:

   - overloads taking an explicit argument of type
     ``parallel_unsequenced_policy`` are introduced into the ``std`` namespace;
   - these will get preferentially selected versus the master template;
   - the body forwards to the equivalent algorithm from the accelerator specific
     library

2. It provides allocation / deallocation functions that are equivalent to the
   standard ones, but obtain memory by invoking
   `hipMallocManaged <https://rocm.docs.amd.com/projects/HIP/en/latest/.doxygen/docBin/html/group___memory_m.html#gab8cfa0e292193fa37e0cc2e4911fa90a>`_
   and release it via `hipFree <https://rocm.docs.amd.com/projects/HIP/en/latest/.doxygen/docBin/html/group___memory.html#ga740d08da65cae1441ba32f8fedb863d1>`_.

Predefined Macros
=================

.. list-table::
   :header-rows: 1

   * - Macro
     - Description
   * - ``__HIPSTDPAR__``
     - Defined when Clang is compiling code in algorithm offload mode, enabled
       with the ``--hipstdpar`` compiler option.
   * - ``__HIPSTDPAR_INTERPOSE_ALLOC__``
     - Defined only when compiling in algorithm offload mode, when the user
       enables interposition mode with the ``--hipstdpar-interpose-alloc``
       compiler option, indicating that all dynamic memory allocation /
       deallocation functions should be replaced with accelerator aware
       variants.

Restrictions
============

We define two modes in which runtime execution can occur:

1. **HMM Mode** - this assumes that the
   `HMM <https://docs.kernel.org/mm/hmm.html>`_ subsystem of the Linux kernel
   is used to provide transparent on-demand paging i.e. memory obtained from a
   system / OS allocator such as via a call to ``malloc`` or ``operator new`` is
   directly accessible to the accelerator and it follows the C++ memory model;
2. **Interposition Mode** - this is a fallback mode for cases where transparent
   on-demand paging is unavailable (e.g. in the Windows OS), which means that
   memory must be allocated via an accelerator aware mechanism, and system
   allocated memory is inaccessible for the accelerator.

The following restrictions imposed on user code apply to both modes:

1. Pointers to function, and all associated features, such as e.g. dynamic
   polymorphism, cannot be used (directly or transitively) by the user provided
   callable passed to an algorithm invocation;
2. Global / namespace scope / ``static`` / ``thread`` storage duration variables
   cannot be used (directly or transitively) in name by the user provided
   callable;

   - When executing in **HMM Mode** they can be used in address e.g.:

     .. code-block:: C++

        namespace { int foo = 42; }

        bool never(const std::vector<int>& v) {
          return std::any_of(std::execution::par_unseq, std::cbegin(v), std::cend(v), [](auto&& x) {
            return x == foo;
          });
        }

        bool only_in_hmm_mode(const std::vector<int>& v) {
          return std::any_of(std::execution::par_unseq, std::cbegin(v), std::cend(v),
                             [p = &foo](auto&& x) { return x == *p; });
        }

3. Only algorithms that are invoked with the ``parallel_unsequenced_policy`` are
   candidates for offload;
4. Only algorithms that are invoked with iterator arguments that model
   `random_access_iterator <https://en.cppreference.com/w/cpp/iterator/random_access_iterator>`_
   are candidates for offload;
5. `Exceptions <https://en.cppreference.com/w/cpp/language/exceptions>`_ cannot
   be used by the user provided callable;
6. Dynamic memory allocation (e.g. ``operator new``) cannot be used by the user
   provided callable;
7. Selective offload is not possible i.e. it is not possible to indicate that
   only some algorithms invoked with the ``parallel_unsequenced_policy`` are to
   be executed on the accelerator.

In addition to the above, using **Interposition Mode** imposes the following
additional restrictions:

1. All code that is expected to interoperate has to be recompiled with the
   ``--hipstdpar-interpose-alloc`` flag i.e. it is not safe to compose libraries
   that have been independently compiled;
2. automatic storage duration (i.e. stack allocated) variables cannot be used
   (directly or transitively) by the user provided callable e.g.

   .. code-block:: c++

      bool never(const std::vector<int>& v, int n) {
        return std::any_of(std::execution::par_unseq, std::cbegin(v), std::cend(v),
                           [p = &n](auto&& x) { return x == *p; });
      }

Current Support
===============

At the moment, C++ Standard Parallelism Offload is only available for AMD GPUs,
when the `ROCm <https://rocm.docs.amd.com/en/latest/>`_ stack is used, on the
Linux operating system. Support is synthesised in the following table:

.. list-table::
   :header-rows: 1

   * - `Processor <https://llvm.org/docs/AMDGPUUsage.html#amdgpu-processors>`_
     - HMM Mode
     - Interposition Mode
   * - GCN GFX9 (Vega)
     - YES
     - YES
   * - GCN GFX10.1 (RDNA 1)
     - *NO*
     - YES
   * - GCN GFX10.3 (RDNA 2)
     - *NO*
     - YES
   * - GCN GFX11 (RDNA 3)
     - *NO*
     - YES
   * - GCN GFX12 (RDNA 4)
     - *NO*
     - YES

The minimum Linux kernel version for running in HMM mode is 6.4.

The forwarding header can be obtained from
`its GitHub repository <https://github.com/ROCmSoftwarePlatform/roc-stdpar>`_.
It will be packaged with a future `ROCm <https://rocm.docs.amd.com/en/latest/>`_
release. Because accelerated algorithms are provided via
`rocThrust <https://rocm.docs.amd.com/projects/rocThrust/en/latest/>`_, a
transitive dependency on
`rocPrim <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/>`_ exists. Both
can be obtained either by installing their associated components of the
`ROCm <https://rocm.docs.amd.com/en/latest/>`_ stack, or from their respective
repositories. The list algorithms that can be offloaded is available
`here <https://github.com/ROCmSoftwarePlatform/roc-stdpar#algorithm-support-status>`_.

HIP Specific Elements
---------------------

1. There is no defined interop with the
   `HIP kernel language <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html>`_;
   whilst things like using `__device__` annotations might accidentally "work",
   they are not guaranteed to, and thus cannot be relied upon by user code;

   - A consequence of the above is that both bitcode linking and linking
     relocatable object files will "work", but it is not guaranteed to remain
     working or actively tested at the moment; this restriction might be relaxed
     in the future.

2. Combining explicit HIP, CUDA or OpenMP Offload compilation with
   ``--hipstdpar`` based offloading is not allowed or supported in any way.
3. There is no way to target different accelerators via a standard algorithm
   invocation (`this might be addressed in future C++ standards <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2500r1.html>`_);
   an unsafe (per the point above) way of achieving this is to spawn new threads
   and invoke the `hipSetDevice <https://rocm.docs.amd.com/projects/HIP/en/latest/.doxygen/docBin/html/group___device.html#ga43c1e7f15925eeb762195ccb5e063eae>`_
   interface e.g.:

   .. code-block:: c++

      int accelerator_0 = ...;
      int accelerator_1 = ...;

      bool multiple_accelerators(const std::vector<int>& u, const std::vector<int>& v) {
        std::atomic<unsigned int> r{0u};

        thread t0{[&]() {
          hipSetDevice(accelerator_0);

          r += std::count(std::execution::par_unseq, std::cbegin(u), std::cend(u), 42);
        }};
        thread t1{[&]() {
          hitSetDevice(accelerator_1);

          r += std::count(std::execution::par_unseq, std::cbegin(v), std::cend(v), 314152)
        }};

        t0.join();
        t1.join();

        return r;
      }

   Note that this is a temporary, unsafe workaround for a deficiency in the C++
   Standard.

Open Questions / Future Developments
====================================

1. The restriction on the use of global / namespace scope / ``static`` /
   ``thread`` storage duration variables in offloaded algorithms will be lifted
   in the future, when running in **HMM Mode**;
2. The restriction on the use of dynamic memory allocation in offloaded
   algorithms will be lifted in the future.
3. The restriction on the use of pointers to function, and associated features
   such as dynamic polymorphism might be lifted in the future, when running in
   **HMM Mode**;
4. Offload support might be extended to cases where the ``parallel_policy`` is
   used for some or all targets.

SPIR-V Support on HIPAMD ToolChain
==================================

The HIPAMD ToolChain supports targetting
`AMDGCN Flavoured SPIR-V <https://llvm.org/docs/SPIRVUsage.html#target-triples>`_.
The support for SPIR-V in the ROCm and HIPAMD ToolChain is under active
development.

Compilation Process
-------------------

When compiling HIP programs with the intent of utilizing SPIR-V, the process
diverges from the traditional compilation flow:

Using ``--offload-arch=amdgcnspirv``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Target Triple**: The ``--offload-arch=amdgcnspirv`` flag instructs the
  compiler to use the target triple ``spirv64-amd-amdhsa``. This approach does
  generates generic AMDGCN SPIR-V which retains architecture specific elements
  without hardcoding them, thus allowing for optimal target specific code to be
  generated at run time, when the concrete target is known.

- **LLVM IR Translation**: The program is compiled to LLVM Intermediate
  Representation (IR), which is subsequently translated into SPIR-V. In the
  future, this translation step will be replaced by direct SPIR-V emission via
  the SPIR-V Back-end.

- **Clang Offload Bundler**: The resulting SPIR-V is embedded in the Clang
  offload bundler with the bundle ID ``hip-spirv64-amd-amdhsa--amdgcnspirv``.

Architecture Specific Macros
----------------------------

None of the architecture specific :doc:`AMDGPU macros <AMDGPUSupport>` are
defined when targeting SPIR-V. An alternative, more flexible mechanism to enable
doing per target / per feature code selection will be added in the future.
