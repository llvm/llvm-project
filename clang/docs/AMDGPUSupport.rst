.. raw:: html

  <style type="text/css">
    .none { background-color: #FFCCCC }
    .part { background-color: #FFFF99 }
    .good { background-color: #CCFF99 }
  </style>

.. role:: none
.. role:: part
.. role:: good

.. contents::
   :local:

==============
AMDGPU Support
==============

Clang supports OpenCL, HIP and OpenMP on AMD GPU targets.


Predefined Macros
=================


.. list-table::
   :header-rows: 1

   * - Macro
     - Description
   * - ``__AMDGPU__``
     - Indicates that the code is being compiled for an AMD GPU.
   * - ``__AMDGCN__``
     - Defined if the GPU target is AMDGCN.
   * - ``__R600__``
     - Defined if the GPU target is R600.
   * - ``__<ArchName>__``
     - Defined with the name of the architecture (e.g., ``__gfx906__`` for the gfx906 architecture).
   * - ``__<GFXN>__``
     - Defines the GFX family (e.g., for gfx906, this macro would be ``__GFX9__``).
   * - ``__amdgcn_processor__``
     - Defined with the processor name as a string (e.g., ``"gfx906"``).
   * - ``__amdgcn_target_id__``
     - Defined with the target ID as a string.
   * - ``__amdgcn_feature_<feature-name>__``
     - Defined for each supported target feature. The value is 1 if the feature is enabled and 0 if it is disabled. Allowed feature names are sramecc and xnack.
   * - ``__AMDGCN_CUMODE__``
     - Defined as 1 if the CU mode is enabled and 0 if the WGP mode is enabled.
   * - ``__AMDGCN_UNSAFE_FP_ATOMICS__``
     - Defined if unsafe floating-point atomics are allowed.
   * - ``__HAS_FMAF__``
     - Defined if FMAF instruction is available (deprecated).
   * - ``__HAS_LDEXPF__``
     - Defined if LDEXPF instruction is available (deprecated).
   * - ``__HAS_FP64__``
     - Defined if FP64 instruction is available (deprecated).

Please note that the specific architecture and feature names will vary depending on the GPU. Also, some macros are deprecated and may be removed in future releases.

AMDGPU Builtins
===============

Clang provides a set of builtins to access low-level, AMDGPU-specific hardware
features directly from C, C++, OpenCL C, and HIP. These builtins often map
directly to a single machine instruction.

.. _builtin-amdgcn-ds-bpermute:

``__builtin_amdgcn_ds_bpermute``
--------------------------------

Performs a backward (pull) permutation of values within a wavefront. This builtin compiles to the
``ds_bpermute_b32`` instruction and implements a "read from lane" semantic using a **byte-based**
address.

**Syntax**

.. code-block:: c++

  T __builtin_amdgcn_ds_bpermute(int index, T src);

**Summary**

All active lanes in the current wavefront conceptually place their ``src`` payloads into an
internal cross-lane buffer. Each lane then reads a 32-bit value from that buffer at the byte
offset given by ``index`` and returns it as type ``T``. The exchange uses LDS hardware paths
but does not access user-visible LDS or imply any synchronization.

This builtin is **polymorphic**: the type of ``src`` determines the return type.

Availability
------------

- Targets: AMD GCN3 (gfx8) and newer.

Parameters
----------

- ``index`` (``int``): Byte offset used to select the source lane. Hardware only consumes bits
  ``[7:2]``. To read the 32-bit value from lane *i*, pass ``i * 4`` as the index.
  Indices that select lanes outside the current wave size or lanes that are inactive at the call
  site yield an unspecified value (commonly zero on current hardware).

- ``src`` (``T``): The value contributed by the current lane. Supported T includes scalar
  integers and floating point types, pointers, vectors, aggregates (structs/unions/complex),
  and trivially copyable C++ classes. The operation is purely bit-wise and trivially copies
  memory representations; it does not invoke or respect C++ copy/move constructors or
  assignment operators. For types whose total size is not a multiple of 32 bits, the
  value is zero-extended to the next multiple of 32 bits before permutation.
  The value is then split into 32-bit words, each independently permuted via
  ``ds_bpermute_b32`` using the same index, and the result is reassembled and
  truncated back to the original bit width before being returned as type ``T``.

Semantics and Guarantees
------------------------

* **Active lane participation**: Only lanes active in the EXEC mask at the call site
  contribute a payload. Reading from an inactive source lane produces an unspecified value.

* **Index per lane**: ``index`` may vary across lanes. Only bits ``[7:2]`` are used for lane
  selection. Bits outside this range are ignored by hardware.

* **No synchronization**: The builtin does not synchronize lanes or waves and does not
  order memory operations. It doesn't read or write user-visible LDS.

* **Wave size**: Valid source lanes are ``0 .. warpSize-1`` (use ``warpSize``/equivalent to
  query 32 vs 64). Selecting lanes outside that range yields an unspecified value.

Examples
--------

Reverse within a wavefront (handles wave32 or wave64):

.. code-block:: c++

  #include <hip/hip_runtime.h>

  __global__ void wavefront_reverse(float* data, int n) {
    int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int lane    = threadIdx.x % warpSize;            // works for 32 or 64
    int peer    = (warpSize - 1) - lane;             // reversed lane
    int offset  = peer * 4;                          // byte address

    float my_val      = data[tid];
    float reversed    = __builtin_amdgcn_ds_bpermute(offset, my_val);
    data[tid] = reversed;
  }
