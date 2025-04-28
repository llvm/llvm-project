.. meta::
  :description: HIPCC environment variables
  :keywords: HIPCC, ROCm, HIP tools, HIP compiler

.. _hipcc_vars:

******************************************
HIPCC environment variables
******************************************

This topic provides descriptions of the HIPCC environment
variables. For more information about other ROCm environment variables, see 
:ref:`ROCm environment variables page <env-variables-reference>`. 

.. list-table::
    :header-rows: 1
    :widths: 50,50

    * - Environment variable
      - Value

    * - | ``HIP_PLATFORM``
        | The platform targeted by HIP. If ``HIP_PLATFORM`` isn't set, then :doc:`HIPCC <hipcc:index>` attempts to auto-detect the platform based on whether the ``nvcc`` tool is found.
      - ``amd``, ``nvidia``

    * - | ``HIP_PATH``
        | The path of the HIP SDK on Microsoft Windows for AMD platforms.
      - Default: ``C:/hip``

    * - | ``ROCM_PATH``
        | The path of the installed ROCm software stack on Linux for AMD platforms.
      - Default: ``/opt/rocm``

    * - | ``CUDA_PATH``
        | Path to the CUDA SDK, which is only used for NVIDIA platforms.
      - Default: ``/usr/local/cuda``

    * - | ``HIP_CLANG_PATH``
        | Path to the clang, which is only used for AMD platforms.
      - Default: ``ROCM_PATH/llvm/bin`` or ``HIP_PATH/../llvm/bin"``

    * - | ``HIP_LIB_PATH``
        | The HIP device library installation path.
      - Default: ``HIP_PATH/lib``

    * - | ``HIP_DEVICE_LIB_PATH``
        | The HIP device library installation path.
      -

    * - | ``HIPCC_COMPILE_FLAGS_APPEND``
        | Append extra flags as compilation options to ``hipcc``.
      -

    * - | ``HIPCC_LINK_FLAGS_APPEND``
        | Append extra flags as compilation options to ``hipcc``.
      -

    * - | ``HIPCC_VERBOSE``
        | Outputs detailed information on subcommands executed during compilation.
      - | 1: Displays the command to ``clang++`` or ``nvcc`` with all options (``hipcc-cmd``).
        | 2: Displays all relevant environment variables and their values.
        | 4: Displays only the arguments passed to the ``hipcc`` command (``hipcc_args``).
        | 5: Displays both the command to ``clang++`` or ``nvcc`` and ``hipcc`` arguments (``hipcc-cmd`` and ``hipcc-args``).
        | 6: Displays all relevant environment variables and their values, along with the arguments to the ``hipcc`` command.
        | 7: Displays all of the above: ``hipcc-cmd``, ``hipcc-args``, and environment variables.
