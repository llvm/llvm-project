.. meta::
  :description: Learn more about the AMD ROCm LLVM compiler infrastructure and its various components and tools, including the open-source ROCm LLVM fork and associated compilers.
  :keywords: compiler, rocm-llvm, ROCm, LLVM, ROCMCC, amdclang++, Clang, hipcc, openmp

************************************************
ROCm LLVM compiler infrastructure
************************************************

The AMD ``llvm-project`` is a fork of `<https://github.com/llvm/llvm-project>`_. The AMD code is open and hosted at `<https://github.com/ROCm/llvm-project>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Conceptual

    * :doc:`Using AddressSanitizer <conceptual/using-gpu-sanitizer>`
    * :doc:`OpenMP support <conceptual/openmp>`
    * :ref:`generic-code`
    * :ref:`spirv`

  .. grid-item-card:: Reference

    * :doc:`AMD ROCm compiler reference <reference/rocmcc>`
    * `Clang documentation <./LLVM/clang/html/index.html>`_
    * `Clang-tools documentation <./LLVM/clang-tools/html/index.html>`_
    * :doc:`HIPCC documentation <hipcc:index>`
    * :doc:`HIPIFY documentation <hipify:index>`
    * `LLD documentation <./LLVM/lld/html/index.html>`_
    * `LLVM documentation <./LLVM/llvm/html/index.html>`_

ROCm includes multiple compilers of varying origins and purposes as described in the following table: 

.. # COMMENT: The following lines define a break for use in the table below. 
.. |br| raw:: html 

    <br />

.. list-table::
    :widths: 2 5

    * - **Name**
      - **Description**

    * - ``amdclang++`` 
      - Clang/LLVM-based compiler that is part of ``rocm-llvm`` package. The source code is available at `<https://github.com/ROCm/llvm-project>`_. 

    * - ``AOCC`` 
      - Closed-source clang-based compiler that includes additional CPU optimizations. **NOTE:** ``AOCC`` is not delivered as part of ROCm. For more information, see `https://developer.amd.com/amd-aocc <https://developer.amd.com/amd-aocc>`_.   

    * - ``HIP-Clang`` 
      - Another name for the ``amdclang++`` compiler.

    * - ``HIPIFY`` 
      - Tools used to automatically translate CUDA source code into portable HIP C++, including ``hipify-clang`` and ``hipify-perl``. The source code is available at `<https://github.com/ROCm/HIPIFY>`__.

    * - ``hipcc`` 
      - HIP compiler driver utility that invokes ``clang`` or ``nvcc`` and passes the appropriate include and library options for the target compiler and HIP infrastructure. See the `<https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc>`_ for more information.

AMD ROCm also provides additional open-source utilities and libraries for building GPU code located in the ``llvm-project/amd`` directory:

.. list-table::
    :widths: 2 5

    * - **Name**
      - **Description**

    * - ``amd/comgr``
      - The Code Object Manager API, designed to simplify linking, compiling, and inspecting code objects. See the `llvm-project/amd/comgr/README <https://github.com/ROCm/llvm-project/tree/amd-staging/amd/comgr>`_ for more information.

    * - ``amd/device-libs``
      - The sources and CMake build system for a set of AMD-specific device-side language runtime libraries. See the `llvm-project/amd/device-libs/README <https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs>`_ for more information.
