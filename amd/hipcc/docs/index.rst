.. meta::
  :description: HIPCC command
  :keywords: HIPCC, ROCm, HIP tools, HIP compiler

.. _hipcc-docs:

******************************************
HIPCC documentation
******************************************

.. note::
  ROCm provides and supports multiple compilers as described in `ROCm compiler reference <https://rocm.docs.amd.com/projects/llvm-project/en/latest/reference/rocmcc.html>`_.

``hipcc`` is a compiler driver utility that will call ``clang`` or ``nvcc``, depending on target, and pass the appropriate include and library options for the target compiler and HIP infrastructure. C++ executable versions of ``hipcc`` and ``hipconfig`` compiler driver utilities are provided.

The HIPCC public repository is located at `https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc <https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc>`_

The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :ref:`hipcc_build`
    * :ref:`hipcc_vars`

  .. grid-item-card:: How to

    * :ref:`hipcc_use`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
