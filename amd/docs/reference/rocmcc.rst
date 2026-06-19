.. meta::
  :description: Learn more about the AMD ROCm LLVM compiler infrastructure and its various components and tools, including the open-source ROCm LLVM fork and associated compilers.
  :keywords: compiler, rocm-llvm, ROCm, LLVM, ROCMCC, amdclang++, Clang, hipcc, openmp

************************************************
ROCm compiler reference
************************************************

ROCm includes compilers optimized for high-performance computing on AMD GPUs and CPUs supporting various heterogeneous programming models such as Heterogeneous-computing Interface for Portability (HIP), OpenMP, and OpenCL. 

.. important:: 
  The ROCm compilers only build the x86 and AMDGPU targets. Functionality and options that are relevant to other targets are not included within the ROCm compilers.

For more details, see:

* `LLVM User Guide for AMD GPU Backend <https://llvm.org/docs/AMDGPUUsage.html>`_
* Releases and source: `https://github.com/ROCm/llvm-project <https://github.com/ROCm/llvm-project>`_

ROCm compiler interfaces
========================

ROCm provides two compiler interfaces for compiling HIP programs:

* ``/opt/rocm/bin/amdclang++``
* ``/opt/rocm/bin/hipcc``

The ROCm compilers leverage the same LLVM compiler technology with the AMD GCN GPU support;
however, they offer a slightly different user experience. The ``hipcc`` command-line
interface provides a more familiar user interface to users who are
experienced in CUDA but relatively new to the ROCm/HIP development environment.
On the other hand, ``amdclang++`` provides a user interface identical to the ``clang++``
compiler. It is more suitable for experienced developers who want to directly
interact with the clang compiler and gain full control of the application
build process.

The major differences between ``hipcc`` and ``amdclang++`` are listed below:

.. # COMMENT: The following lines define a break for use in the table below. 
.. |br| raw:: html 

    <br />

.. csv-table::
  :widths: 20, 40, 40
  :header: Feature, ``hipcc``, ``amdclang++``

  Compiling HIP source files, Treats ``.hip`` source files as HIP language source files, "Enables the HIP language support for files with the ``.hip`` extension or through the ``-x hip`` compiler option"
  "Detecting GPU architecture", "Auto-detects the GPUs available on the system and generates code for those devices when no GPU architecture is specified", "Has AMD GCN gfx803 as the default GPU architecture. The ``--offload-arch`` compiler option may be used to target other GPU architectures"
  "Finding a HIP installation", "Finds the HIP installation based on its own location and its knowledge about the ROCm directory structure", "First looks for HIP under the same parent directory as its own LLVM directory and then falls back on ``/opt/rocm``. Users can use the ``--rocm-path`` option to instruct the compiler to use HIP from the specified ROCm installation."
  "Linking to the HIP runtime library", "Is configured to automatically link to the HIP runtime from the detected HIP installation", "Requires the ``--hip-link`` flag to be specified to link to the HIP runtime. Alternatively, users can use the ``-l<dir>  - Lamdhip64`` option to link to a HIP runtime library."
  "Source code location:", `ROCm/llvm-project/amd/hipcc <https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc>`_, `ROCm/llvm-project/clang <https://github.com/ROCm/llvm-project/tree/amd-staging/clang>`_

Compiler options and features
=============================

This section discusses compiler options and features.

AMD GPU compilation
-------------------

This section outlines commonly used compiler flags for ``hipcc`` and ``amdclang++``.

.. list-table::
    :widths: 2 5

    * - **Options**
      - **Description**

    * - ``-x hip``
      - Compiles the source file as a HIP program.

    * - ``-fopenmp``
      - Enables the OpenMP support.

    * - ``-fopenmp-targets=<gpu>``
      - Enables the OpenMP target offload support of the specified GPU architecture, where ``<gpu>`` specifies the GPU architecture. For example: gfx908

    * - ``--gpu-max-threads-per-block=<value>:``
      - Sets the default limit of threads per block, also referred to as the launch bounds, where ``<value>`` specifies the default maximum amount of threads per block.

    * - ``-munsafe-fp-atomics``
      - Enables unsafe floating point atomic instructions (AMDGPU only).

    * - ``-ffast-math``
      - Allows aggressive, lossy floating-point optimizations.

    * - ``-mwavefrontsize64, -mno-wavefrontsize64``
      - Sets wavefront size to be 64 or 32 on RDNA architectures.

    * - ``-mcumode``
      - Switches between CU and WGP modes on RDNA architectures.

    * - ``--offload-arch=<gpu>``
      - HIP offloading target ID. |br| May be specified more than once, where ``<gpu>`` specifies the device architecture followed by target ID features delimited by a colon. |br| Each target ID feature is a predefined string followed by a plus or minus sign (e.g. ``gfx908:xnack+:sramecc-``).

    * - ``-g``
      - Generates source-level debug information.

    * - ``-fgpu-rdc,    -fno-gpu-rdc``
      - Generates relocatable device code, also known as separate compilation mode.

    * - ``-foffload-lto=<arg>``
      - Link Time Optimization (LTO) is a compiler optimization technique that performs program-wide analysis and optimization at the linking stage, enabling more aggressive and comprehensive optimizations than those possible at the file level. The ``-foffload-lto`` option for Clang is associated with enabling LTO for offload targets. ThinLTO reduces the memory and time overhead associated with full LTO while still providing significant performance improvements. ThinLTO does not work for all code, so it is not enabled by default. If the device code contains any form of indirect function call (including C++ virtual functions), it is not expected to work. To enable full LTO for device code, use the ``-foffload-lto=full`` option, or simply ``-foffload-lto``, for both the compile and linker flags. To enable ThinLTO, use ``-foffload-lto=thin`` for both the compile and linker flags. You can separately specify ``-flto`` for CPU LTO, and ``-foffload-lto`` for device LTO. For more information, see `Clang Reference <https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/clang/html/ClangCommandLineReference.html#target-independent-compilation-options>`__.

    * - ``-flto-partitions=<num>``
      - Equivalent to the ``--lto-partitions=<num>`` LLD option. Controls the number of partitions used for parallel code generation when using full LTO (including when using ``-fgpu-rdc``). The number of partitions must be greater than 0, and a value of 1 disables the feature. The default value is 8. For more information, see :ref:`parallel-code`.

.. _parallel-code:

Parallel code generation
------------------------

The compiler driver will use parallel code generation by default when compiling using
full LTO (including when using the ``-fgpu-rdc`` option) for HIP. This divides the optimized
LLVM IR module into roughly equal partitions before instruction selection and lowering,
which can help improve build times.

Each kernel in the linked LTO module can be put in a separate partition, and any non-inlined
function it depends on can be copied alongside it. Thus, while parallel code generation can
improve build time, it can also duplicate non-inlined, non-kernel functions across multiple partitions,
potentially increasing the binary size of the final object file.

Developers are encouraged to experiment with different numbers of partitions using
the ``-flto-partitions`` Clang command line option. Recommended values are 1 to 16 partitions,
with especially large projects containing many kernels potentially benefitting from up to 64
partitions. It is not recommended to use a value greater than the number of threads on the
machine. Smaller projects, or projects that contain only a few kernels might also not benefit
at all from partitioning and could even see a slight increase in build time due to the small
overhead of analyzing and partitioning the modules.

Controlling atomic code generation
----------------------------------

Generating code for atomic operations with AMDGPU targets can have complications related to accessing device/remote memory. You can control how atomic operations are lowered in LLVM IR by using the ``[[clang::atomic]]`` attribute. This attribute lets you provide additional metadata to the backend to specify options for atomic operations. For example, you can indicate that certain atomic operations are compatible with specific types of memory, or ensure correctness in denormal mode for floating-point operations. For additional information, see `Extensions for controlling atomic code generation <https://clang.llvm.org/docs/LanguageExtensions.html#extensions-for-controlling-atomic-code-generation>`__ in the Clang documentation.

Inline ASM statements
---------------------

Inline assembly (ASM) statements allow a developer to include assembly
instructions directly in either host or device code. The ROCm compiler
supports ASM statements, however you should not use them for the following reasons:

* The compiler's ability to produce both correct code and optimize surrounding code is impeded.
* The compiler does not parse the content of the ASM statements and cannot examine its contents.
* The compiler must make conservative assumptions in an effort to retain correctness.
* The conservative assumptions may yield code that is less performant compared to code without ASM statements. It is possible that a syntactically correct ASM statement may cause incorrect runtime behavior.
* ASM statements are often ASIC-specific; code containing them is less portable and adds a maintenance burden for the developer if different ASICs are targeted.
* Writing correct ASM statements is often difficult; thorough testing of any ASM statements is strongly encouraged.

.. note::

  For developers who include ASM statements in the code, AMD is interested in understanding the use case and appreciates your feedback at 
  `https://github.com/ROCm/ROCm/issues <https://github.com/ROCm/ROCm/issues>`_


Miscellaneous OpenMP compiler features
--------------------------------------

This section discusses features that have been added or enhanced in the OpenMP
compiler.

Offload-arch tool
^^^^^^^^^^^^^^^^^

An LLVM library and tool that is used to query the execution capability of the
current system as well as to query requirements of a binary file. It is used by
OpenMP device runtime to ensure compatibility of an image with the current
system while loading it. It is compatible with target ID support and multi-image
fat binary support.

**Usage:**

.. code-block:: shell

  offload-arch [Options] [Optional lookup-value]

When used without an option, ``offload-arch`` prints the value of the first offload
architecture found in the underlying system. This can be used by various clang
front ends. For example, to compile for OpenMP offloading on your current system,
invoke clang with the following command:

.. code-block:: shell

  clang -fopenmp  -fopenmp-targets=``offload-arch`` foo.c

If an optional lookup-value is specified, ``offload-arch`` will check if the value
is either a valid ``offload-arch`` or a codename and look up requested additional
information.

The following command provides all the information for offload-arch gfx906:

.. code-block:: shell

  offload-arch gfx906  - V


The options are listed below:

.. list-table::
    :widths: 1 5

    * - **Options**
      - **Description**

    * - ``-h``
      - Prints the help message.

    * - ``-a``
      - Prints values for all devices. Do not stop at the first device found.

    * - ``-m``
      - Prints device code name (often found in ``pci.ids`` file).

    * - ``-n``
      - Prints numeric ``pci-id``.

    * - ``-t``
      - Prints clang offload triple to use for the offload arch.

    * - ``-v``
      - Verbose. Implies the following options ``-a -m -n -t``

    * - ``-f <file>``
      - Prints offload requirements including offload-arch for each compiled offload image built into an application binary file.

    * - ``-c``
      - Prints offload capabilities of the underlying system. This option is used by the language runtime to select an image when multiple images are available. A capability must exist for each requirement of the selected image.

There are symbolic link aliases ``amdgpu-offload-arch`` and ``nvidia-arch`` for
``offload-arch``. These aliases return 1 if no AMD GCN GPU or CUDA GPU is found.
These aliases are useful in determining whether architecture-specific tests
should be run or to conditionally load architecture-specific software.

Command-line simplification using offload-arch flag
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Legacy mechanism of specifying offloading target for OpenMP involves using three
flags, ``-fopenmp-targets``, ``-Xopenmp-target``, and ``-march``. The first two flags
take a target triple (like ``amdgcn-amd-amdhsa`` or ``nvptx64-nvidia-cuda``), while
the last flag takes device name (like ``gfx908`` or ``sm_70``) as input.

Alternatively, users of the ROCm compiler can simply use the flag ``-offload-arch`` for a
combined effect of the preceding three flags.

**Example:**

.. code-block:: shell

  # Legacy mechanism
  clang -fopenmp -target x86_64-linux-gnu \
  -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa \
  -march=gfx906 helloworld.c -o helloworld

**Example:**

.. code-block:: shell

  # Using offload-arch flag
  clang -fopenmp -target x86_64-linux-gnu \
  --offload-arch=gfx906 helloworld.c -o helloworld.

To ensure backward compatibility, both styles are supported. This option is
compatible with target ID support and multi-image fat binaries.

Target ID support for OpenMP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ROCm compiler supports specification of target features along with the GPU
name while specifying a target offload device in the command line, using
``-march`` or ``--offload-arch`` options. The compiled image in such cases is
specialized for a given configuration of device and target features (target ID).

**Example:**

.. code-block:: shell

  # compiling for a gfx908 device with XNACK paging support turned ON
  clang -fopenmp -target x86_64-linux-gnu \
  -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa \
  -march=gfx908:xnack+ helloworld.c -o helloworld

**Example:**

.. code-block:: shell

  # compiling for a gfx908 device with SRAMECC support turned OFF
  clang -fopenmp -target x86_64-linux-gnu \
  -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa \
  -march=gfx908:sramecc- helloworld.c -o helloworld

**Example:**

.. code-block:: shell

  # compiling for a gfx908 device with SRAMECC support turned ON and XNACK paging support turned OFF
  clang -fopenmp -target x86_64-linux-gnu \
  -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa \
  -march=gfx908:sramecc+:xnack- helloworld.c -o helloworld

The target ID specified on the command line is passed to the clang driver using
``target-feature`` flag, to the LLVM optimizer and back end using ``-mattr`` flag, and
to linker using ``-plugin-opt=-mattr`` flag. This feature is compatible with
``offload-arch`` command-line option and multi-image binaries for multiple
architectures.

Multi-image fat binary for OpenMP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ROCm compiler is enhanced to generate binaries that can contain
heterogenous images. This heterogeneity could be in terms of:

* Images of different architectures, like AMD GCN and NVPTX
* Images of same architectures but for different GPUs, like gfx906 and gfx908
* Images of same architecture and same GPU but for different target features,
  like ``gfx908:xnack+`` and ``gfx908:xnack-``

An appropriate image is selected by the OpenMP device runtime for execution
depending on the capability of the current system. This feature is compatible
with target ID support and ``offload-arch`` command-line options and uses
offload-arch tool to determine capability of the current system.

**Example:**

.. code-block:: shell

  clang -fopenmp -target x86_64-linux-gnu \
  -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
  -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 \
  -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 \
  helloworld.c -o helloworld

**Example:**

.. code-block:: shell

  clang -fopenmp -target x86_64-linux-gnu \
  --offload-arch=gfx906 \
  --offload-arch=gfx908 \
  helloworld.c -o helloworld

**Example:**

.. code-block:: shell

  clang -fopenmp -target x86_64-linux-gnu \
  -fopenmp-targets=amdgcn-amd-amdhsa,amdgcn-amd-amdhsa,amdgcn-amd-amdhsa,amdgcn-amd-amdhsa \
  -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:sramecc+:xnack+ \
  -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:sramecc-:xnack+ \
  -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:sramecc+:xnack- \
  -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908:sramecc-:xnack- \
  helloworld.c -o helloworld

ROCmCC compilers create an instance of toolchain for each unique combination
of target triple and the target GPU (along with the associated target features).
``clang-offload-wrapper`` tool is modified to insert a new structure
``__tgt_image_info`` along with each image in the binary. Device runtime is also
modified to query this structure to identify a compatible image based on the
capability of the current system.

Unified shared memory
^^^^^^^^^^^^^^^^^^^^^

The following OpenMP pragma is available on MI200, and it must be executed with
``xnack+`` support.

.. code-block:: shell

  omp requires unified_shared_memory

For more details on unified shared memory, see :doc:`OpenMP support <../conceptual/openmp>`.
