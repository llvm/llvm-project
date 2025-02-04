.. meta::
  :description: HIPCC command
  :keywords: HIPCC, ROCm, HIP tools, HIP compiler

.. _hipcc-docs:

******************************************
HIPCC documentation
******************************************

``hipcc`` is a compiler driver utility that will call clang or nvcc, depending on target, and pass the appropriate include and library options for the target compiler and HIP infrastructure.

There are both C++ and Perl executable versions of the ``hipcc`` and ``hipconfig`` compiler driver utilities provided. By default the C++ version is used when ``hipcc`` is run.

.. note:: 
  You can manually run the Perl scripts using hipcc.pl and hipconfig.pl from the installation. However, you must ensure Perl is installed on the system for the scripts to work. Perl is not automatically installed with the ROCm installation.

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
