.. meta::
  :description: HIPCC environment variables
  :keywords: HIPCC, ROCm, HIP tools, HIP compiler

.. _hipcc_use:

******************************************
Using HIPCC
******************************************

The built executables can be used the same way as the ``hipcc`` and ``hipconfig`` Perl scripts. 
To use the newly built executables from the build folder use ``./`` in front of the executable name. 
For example:

.. code-block:: shell

    ./hipconfig --help
    ./hipcc --help
    ./hipcc --version
    ./hipconfig --full

