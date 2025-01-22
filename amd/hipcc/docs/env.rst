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
* ``HIPCC_COMPILE_FLAGS_APPEND``       : Append extra flags as compilation options to ``hipcc``.
* ``HIPCC_LINK_FLAGS_APPEND``       : Append extra flags as linking options to ``hipcc``.
* ``HIPCC_VERBOSE``  : Outputs detailed information on subcommands executed during compilation.

  - ``HIPCC_VERBOSE = 1``: Displays the command to ``clang++`` or ``nvcc`` with all options (`hipcc-cmd`).
  - ``HIPCC_VERBOSE = 2``: Displays all relevant environment variables and their values.
  - ``HIPCC_VERBOSE = 4``: Displays only the arguments passed to the ``hipcc`` command (`hipcc_args`).
  - ``HIPCC_VERBOSE = 5``: Displays both the command to ``clang++`` or ``nvcc`` and ``hipcc`` arguments (`hipcc-cmd` and `hipcc-args`).
  - ``HIPCC_VERBOSE = 6``: Displays all relevant environment variables and their values, along with the arguments to the ``hipcc`` command.
  - ``HIPCC_VERBOSE = 7``: Displays all of the above: `hipcc-cmd`, `hipcc-args`, and environment variables.
 
                            


