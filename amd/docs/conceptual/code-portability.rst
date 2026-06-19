
.. meta::
  :description: A description of the AMD ROCm LLVM compiler support for AMDGCN flavored SPIR-V.

  :keywords: compiler, rocm-llvm, ROCm, LLVM, SPIR-V, amdclang++, Clang

.. _generic-code:

********************************
Code portability and compression
********************************

SPIR-V, generic code objects, and offload compression are three technologies available in the ROCm software stack. This article describes these technologies and highlights situations when a particular solution might best be used.


SPIR-V
======

SPIR-V is an additional offload architecture, as described in :ref:`spirv`. The SPIR-V architecture is an abstract architecture that can stand in for multiple AMD GPUs. At runtime the abstract target is redefined as the concrete target of a specific AMD GPU, and details of the target become available at runtime. 

To support SPIR-V, developed by the `Khronos Group <http://khronos.org/spirv/>`_, additions have been made to the `compiler <https://clang.llvm.org/docs/HIPSupport.html#spir-v-support-on-hipamd-toolchain>`_ within the ROCm software stack. Also, a fork of the `SPIR-V LLVM Translator <https://github.com/ROCm/SPIRV-LLVM-Translator>`_ has been created and a limited number of out-of-tree modifications have been added.

Generic code objects
====================

Generic code objects offer a mechanism to reduce the number of targeted offload architectures by targeting related "generic" offload architectures. Generic offload architectures reduce binary size and can also reduce build time. For example, instead of building for ``gfx1030`` through ``gfx1036`` (seven targets), you can target ``gfx10-3-generic`` (a single target) and build a binary that will run on each of those seven targets. The `User Guide for AMDGPU Backend <https://llvm.org/docs/AMDGPUUsage.html#user-guide-for-amdgpu-backend>`_ discusses `Generic Processing Versioning <https://llvm.org/docs/AMDGPUUsage.html#generic-processor-versioning>`_ and generic code objects, and lists currently supported `generic offload architectures <https://llvm.org/docs/AMDGPUUsage.html#amdgpu-generic-processor-table>`_.

However, performance might be impacted because a generic code object is the lowest common denominator of all the related targets of the generic code object. Features of one target, such as ``gfx1033``, might be unavailable when building the ``gfx10-3-generic`` code object if related members of that generic code object do not support the feature. Though, in general, generic code objects are defined so that there is no significant performance difference between generic code objects and its related targets. 


Offload compression
===================

Offload compression is a HIP compiler option used to compress the device code. It does not modify the device code. For applications that target multiple offload architectures and want to decrease the overall size of the binary, offload compression is a good option. The feature is invoked via the `--offload-compress <https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-offload-compress>`_ compiler option:


.. code:: bash

  clang --offload-compress ...

Compression happens at compile-time. Decompression occurs on the host at runtime, once for the entire compressed bundle. Compression and decompression are fast operations and do not add significant overhead to the compilation or runtime. Compression ratios will vary depending on the generated code. You can use the environment variable ``OFFLOAD_BUNDLER_VERBOSE=1`` to output the compression ratio. 
