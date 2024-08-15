.. meta::
  :description: HIPCC environment variables
  :keywords: HIPCC, ROCm, HIP tools, HIP compiler

.. _hipcc_vars:

******************************************
HIPCC environment variables
******************************************

The environment variable ``HIP_PLATFORM`` can be used to specify ``amd`` or ``nvidia`` depending on the available backend tool flows:

* ``HIP_PLATFORM`` = ``amd`` or ``HIP_PLATFORM`` = ``nvidia``

.. note:: 
    If ``HIP_PLATFORM`` is not set, then ``hipcc`` will attempt to auto-detect based on if the ``nvcc`` tool is found.

Additional environment variable controls:

* ``CUDA_PATH``       : Path to the CUDA SDK. The default is ``/usr/local/cuda``. This is only used for NVIDIA platforms.
