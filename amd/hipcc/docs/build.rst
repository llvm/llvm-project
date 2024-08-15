.. meta::
  :description: HIPCC environment variables
  :keywords: HIPCC, ROCm, HIP tools, HIP compiler

.. _hipcc_build:

******************************************
Building and testing HIPCC
******************************************

To build the ``hipcc`` and ``hipconfig`` executables, use the following commands. 

.. code-block:: bash

    mkdir build
    cd build

    cmake ..

    make -j

.. note::
  The tools are created in the current build folder, and will need to be copied to ``/opt/rocm/hip/bin`` folder location.

Testing HIPCC
=============

Currently ``hipcc`` and ``hipconfig`` tools are tested by building and running test samples that can be found at `HIP-tests <https://github.com/ROCm/hip-tests/tree/develop/samples>`_. 
