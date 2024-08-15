.. meta::
  :description: HIPCC command
  :keywords: HIPCC, ROCm, HIP tools, HIP compiler

.. _hipcc-docs:

******************************************
HIPCC documentation
******************************************

``hipcc`` is a compiler driver utility that will call clang or nvcc, depending on target, and pass the appropriate include and library options for the target compiler and HIP infrastructure. 

There are both Perl and C++ executable versions of the ``hipcc`` and ``hipconfig`` compiler driver utilities provided. By default the C++ version is used when ``hipcc`` is run. To enable the Perl versions of these commands set the environment variable ``HIP_USE_PERL_SCRIPTS=1``.

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
