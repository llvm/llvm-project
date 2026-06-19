
.. meta::
  :description: A description of the AMD ROCm LLVM compiler support for AMDGCN Flavoured SPIR-V.
  :keywords: compiler, rocm-llvm, ROCm, LLVM, SPIR-V, amdclang++, Clang

.. _spirv:

***********************
ROCm support for SPIR-V
***********************

ROCm provides support for generating portable, AMD GPU target-agnostic `SPIR-V <http://khronos.org/spirv/>`_ from HIP source, by picking the ``amdgcnspirv`` offload architecture. For example:


.. code-block:: bash

    clang++ -x hip --offload-arch=amdgcnspirv main.cpp -o main
    # or
    clang++ -target spirv64-amd-amdhsa -x hip main.cpp -o main

The ``amdgcnspirv`` offload architecture represents AMDGCN SPIR-V, which enables the following AMD specific features on top of the baseline SPIR-V 1.6 capabilities:

* AMDGCN inline ASM is supported
* AMDGCN target-specific builtins are supported
* The feature set matches the union of AMDGCN targets' features

LLVM provides additional details:

* `About HIP and SPIR-V <https://clang.llvm.org/docs/HIPSupport.html#spir-v-support-on-hipamd-toolchain>`_
* `About the SPIR-V target <https://llvm.org/docs/SPIRVUsage.html>`_

Abstract target versus concrete gfx targets
===========================================

The ``amdgcnspirv`` target is abstract. It is not tied to a specific GPU, but can stand in for any AMD GPU. A consequence of the abstract nature of the target is that some information only becomes available at run time, when SPIR-V gets lowered to native code for a concrete GPU:

* The concrete GPU architecture is not established at compile time:

  - The ``__<ArchName>__``, ``__<GFXN>__``, ``__amdgcn_processor__`` and ``__amdgcn_target_id__`` macros are not defined at compilation

* Improved target-specific extensions can be tied to a specific GPU as described in `AMDGPU language extensions <https://github.com/ROCm/llvm-project/blob/c2535466c6e40acd5ecf6ba1676a4e069c6245cc/clang/docs/LanguageExtensions.rst#target-specific-extensions>`_:

  - ``__builtin_amdgcn_processor_is`` for queries of the current target processor.
  - ``__builtin_amdgcn_is_invocable`` enables fine-grained, per-builtin feature availability.

* The physical wavefront size is not available at compile time:

  - ``warpSize`` constant value is not ``constexpr`` / ``consteval``
  - The ``__AMDGCN_WAVEFRONT_SIZE`` and ``__AMDGCN_WAVEFRONT_SIZE__`` macros are not defined at compilation, but these macros are deprecated and should no longer be used.

* Given that an additional run time compilation element is needed in the SPIR-V workflow, extra run time overhead might be observed. You should consider this overhead when measuring the timing and performance of this workflow.

Compatibility with precompiled ROCm libraries
---------------------------------------------

A client application or library that targets SPIR-V can work with precompiled ROCm components with concrete targets. For example, a program `P` which has both its own HIP kernels (``__global__`` functions) and calls to the rocBLAS library can use ``--offload-arch=amdgcnspirv`` without any additional changes to its compilation flow or its set of library dependencies. However, the general effect is that the abstract targets are limited to the concrete targets of the precompiled ROCm library.

