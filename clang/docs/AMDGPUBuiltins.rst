===============
AMDGPU Builtins
===============

.. contents::
   :local:
   :depth: 2

This document describes the AMDGPU target-specific builtins available in Clang.
Most of these builtins provide direct access to AMDGPU hardware instructions
and intrinsics. They are defined in ``clang/include/clang/Basic/BuiltinsAMDGPU.td``
and typically lower to LLVM intrinsics defined in
``llvm/include/llvm/IR/IntrinsicsAMDGPU.td``.

.. warning::

   These builtins, including their names, arguments, and target requirements,
   are all subject to change without warning across LLVM releases.

All AMDGPU builtins use the ``__builtin_amdgcn_`` prefix (or ``__builtin_r600_``
for R600 targets). Arguments marked ``_Constant`` must be compile-time
constant expressions.

ABI / Special Register Builtins
===============================

These builtins provide access to kernel dispatch metadata, work-item and
workgroup identification, and other ABI-level information. They are available
on all SI+ targets.

Pointer Builtins
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Builtin
     - Description
   * - ``void __constant * __builtin_amdgcn_dispatch_ptr()``
     - Returns a pointer (in constant address space 4) to the dispatch packet
       (``hsa_kernel_dispatch_packet_t``). Used internally to derive workgroup
       size, grid size, and other dispatch parameters.
   * - ``void __constant * __builtin_amdgcn_kernarg_segment_ptr()``
     - Returns a pointer to the beginning of the kernel argument segment.
   * - ``void __constant * __builtin_amdgcn_implicitarg_ptr()``
     - Returns a pointer to the implicit arguments appended after explicit
       kernel arguments. Layout depends on the code object version.
   * - ``void __constant * __builtin_amdgcn_queue_ptr()``
     - Returns a pointer to the ``hsa_queue_t`` object for the queue executing
       the current kernel.

Work-Item and Workgroup Identification
--------------------------------------

All of these are ``Const`` (pure) builtins that take no arguments and return
``unsigned int`` (or ``unsigned short`` for workgroup size).

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Builtin
     - Return Type
     - Description
   * - ``__builtin_amdgcn_workgroup_id_{x,y,z}()``
     - ``unsigned int``
     - Workgroup ID in the specified dimension.
   * - ``__builtin_amdgcn_workitem_id_{x,y,z}()``
     - ``unsigned int``
     - Work-item (thread) ID within the workgroup.
   * - ``__builtin_amdgcn_workgroup_size_{x,y,z}()``
     - ``unsigned short``
     - Workgroup size in the specified dimension. Lowered via a load from the
       dispatch or implicit argument pointer, not a dedicated instruction.
   * - ``__builtin_amdgcn_grid_size_{x,y,z}()``
     - ``unsigned int``
     - Total grid size in the specified dimension. Lowered via a load from the
       dispatch pointer.

**GFX1250+ Cluster Identification** (requires ``gfx1250-insts``):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Builtin
     - Description
   * - ``__builtin_amdgcn_cluster_id_{x,y,z}()``
     - Cluster ID in the specified dimension.
   * - ``__builtin_amdgcn_cluster_workgroup_id_{x,y,z}()``
     - Workgroup ID within the cluster.
   * - ``__builtin_amdgcn_cluster_workgroup_flat_id()``
     - Flat (linearized) workgroup ID within the cluster.
   * - ``__builtin_amdgcn_cluster_workgroup_max_id_{x,y,z}()``
     - Maximum workgroup ID within the cluster.
   * - ``__builtin_amdgcn_cluster_workgroup_max_flat_id()``
     - Maximum flat workgroup ID within the cluster.

Other ABI Builtins
------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Builtin
     - Description
   * - ``unsigned int __builtin_amdgcn_mbcnt_lo(unsigned int mask, unsigned int val)``
     - Counts the number of set bits in ``mask`` for lanes lower than the
       current lane within the lower 32 bits of the exec mask, adds ``val``.
   * - ``unsigned int __builtin_amdgcn_mbcnt_hi(unsigned int mask, unsigned int val)``
     - Same as ``mbcnt_lo`` but for the upper 32 bits of the exec mask.
   * - ``uint64_t __builtin_amdgcn_s_memtime()``
     - Returns a 64-bit timestamp counter. Requires ``s-memtime-inst``.

Instruction Builtins
====================

Scalar Instruction Builtins
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Builtin
     - Description
   * - ``unsigned int __builtin_amdgcn_s_getreg(_Constant int hwreg)``
     - Reads a hardware register. ``hwreg`` is an encoded register specifier
       (register ID, offset, and width packed into 16 bits).
   * - ``void __builtin_amdgcn_s_setreg(_Constant int hwreg, unsigned int val)``
     - Writes ``val`` to a hardware register. ``hwreg`` must be in
       range [0, 65535].
   * - ``uint64_t __builtin_amdgcn_s_getpc()``
     - Returns the current program counter.
   * - ``void __builtin_amdgcn_s_waitcnt(_Constant int cnt)``
     - Inserts an ``s_waitcnt`` instruction with the encoded wait count.
   * - ``void __builtin_amdgcn_s_sendmsg(_Constant int msg, unsigned int gsdata)``
     - Sends message ``msg`` with GS data in ``gsdata``.
   * - ``void __builtin_amdgcn_s_sendmsghalt(_Constant int msg, unsigned int gsdata)``
     - Same as ``s_sendmsg`` but also halts the wavefront.
   * - ``void __builtin_amdgcn_s_barrier()``
     - Inserts a workgroup barrier.
   * - ``void __builtin_amdgcn_s_ttracedata(int data)``
     - Writes ``data`` to the thread trace buffer.
   * - ``void __builtin_amdgcn_s_sleep(_Constant int duration)``
     - Sleeps for approximately ``duration`` cycles.
   * - ``void __builtin_amdgcn_s_incperflevel(_Constant int level)``
     - Increments the performance counter level.
   * - ``void __builtin_amdgcn_s_decperflevel(_Constant int level)``
     - Decrements the performance counter level.
   * - ``void __builtin_amdgcn_s_setprio(_Constant short prio)``
     - Sets the wavefront priority.
   * - ``void __builtin_amdgcn_s_dcache_inv()``
     - Invalidates the scalar data cache.
   * - ``void __builtin_amdgcn_buffer_wbinvl1()``
     - Write-back and invalidate L1 buffer cache.
   * - ``unsigned int __builtin_amdgcn_groupstaticsize()``
     - Returns the size of static LDS allocation in the current workgroup.
   * - ``unsigned int __builtin_amdgcn_wavefrontsize()``
     - Returns the wavefront size (32 or 64).
   * - ``void __builtin_amdgcn_wave_barrier()``
     - Inserts a wave-level barrier hint.

Division and Math Builtins
--------------------------

Division Support
^^^^^^^^^^^^^^^^

These builtins implement steps of the iterative double-precision division
algorithm.

``__builtin_amdgcn_div_scale`` / ``__builtin_amdgcn_div_scalef``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   double __builtin_amdgcn_div_scale(double numer, double denom, bool select_quotient, bool *flag_out);
   float  __builtin_amdgcn_div_scalef(float numer, float denom, bool select_quotient, bool *flag_out);

Scales the numerator or denominator for a subsequent iterative division.

- ``numer``: The numerator.
- ``denom``: The denominator.
- ``select_quotient``: If ``true``, selects the numerator for scaling; if
  ``false``, selects the denominator.
- ``flag_out``: Pointer to a ``bool`` where the overflow/underflow flag is
  written.

**Lowering note**: The underlying intrinsic returns ``{result, flag}`` as a
struct. The builtin unpacks this, returning the result and storing the flag
through the pointer.

``__builtin_amdgcn_div_fmas`` / ``__builtin_amdgcn_div_fmasf``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   double __builtin_amdgcn_div_fmas(double a, double b, double c, bool vcc);
   float  __builtin_amdgcn_div_fmasf(float a, float b, float c, bool vcc);

Fused multiply-add for division, with VCC flag input.

- ``a``, ``b``, ``c``: FMA operands (computes ``a * b + c``).
- ``vcc``: The flag from ``div_scale``.

**Lowering note**: The integer ``vcc`` argument is converted to ``i1`` via
``IsNotNull`` before passing to the intrinsic.

``__builtin_amdgcn_div_fixup`` / ``__builtin_amdgcn_div_fixupf`` / ``__builtin_amdgcn_div_fixuph``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   double __builtin_amdgcn_div_fixup(double a, double b, double c);
   float  __builtin_amdgcn_div_fixupf(float a, float b, float c);
   __fp16 __builtin_amdgcn_div_fixuph(__fp16 a, __fp16 b, __fp16 c);  // requires 16-bit-insts

Applies post-division fixup for special values (NaN, Inf, zero).

Trigonometric Pre-operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   double __builtin_amdgcn_trig_preop(double src, int segment);
   float  __builtin_amdgcn_trig_preopf(float src, int segment);

Looks up ``2.0 / pi`` with segment selector ``segment[4:0]`` for range
reduction before trigonometric operations.

Single-Argument Math Builtins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These builtins compute hardware-precision math operations. The ``f32`` versions
(e.g., ``sinf``, ``logf``) may not handle denormals correctly. The ``h``-suffixed
variants require ``16-bit-insts``.

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Operation
     - f64
     - f32
     - f16
   * - Reciprocal
     - ``__builtin_amdgcn_rcp``
     - ``__builtin_amdgcn_rcpf``
     - ``__builtin_amdgcn_rcph``
   * - Square root
     - ``__builtin_amdgcn_sqrt``
     - ``__builtin_amdgcn_sqrtf``
     - ``__builtin_amdgcn_sqrth``
   * - Reciprocal sqrt
     - ``__builtin_amdgcn_rsq``
     - ``__builtin_amdgcn_rsqf``
     - ``__builtin_amdgcn_rsqh``
   * - Reciprocal sqrt clamp
     - ``__builtin_amdgcn_rsq_clamp``
     - ``__builtin_amdgcn_rsq_clampf``
     -
   * - Sine (input: turns)
     -
     - ``__builtin_amdgcn_sinf``
     - ``__builtin_amdgcn_sinh``
   * - Cosine (input: turns)
     -
     - ``__builtin_amdgcn_cosf``
     - ``__builtin_amdgcn_cosh``
   * - Log2
     -
     - ``__builtin_amdgcn_logf``
     -
   * - Log clamp
     -
     - ``__builtin_amdgcn_log_clampf``
     -
   * - Exp2
     -
     - ``__builtin_amdgcn_exp2f``
     -
   * - Fraction
     - ``__builtin_amdgcn_fract``
     - ``__builtin_amdgcn_fractf``
     - ``__builtin_amdgcn_fracth``
   * - Mantissa
     - ``__builtin_amdgcn_frexp_mant``
     - ``__builtin_amdgcn_frexp_mantf``
     - ``__builtin_amdgcn_frexp_manth``
   * - Exponent
     - ``__builtin_amdgcn_frexp_exp``
     - ``__builtin_amdgcn_frexp_expf``
     - ``__builtin_amdgcn_frexp_exph``

Note: ``sinf``/``cosf`` take input in **turns** (1.0 = full circle), not
radians. ``logf`` performs ``log2``. ``exp2f`` performs ``2^x``. The ``frexp_exp``
variants return ``int`` (or ``short`` for f16).

Ldexp
^^^^^

.. code-block:: c

   double __builtin_amdgcn_ldexp(double x, int exp);
   float  __builtin_amdgcn_ldexpf(float x, int exp);
   __fp16 __builtin_amdgcn_ldexph(__fp16 x, int exp);  // requires 16-bit-insts

Computes ``x * 2^exp``. Lowered to the standard ``llvm.ldexp`` intrinsic.
For the ``h`` variant, the exponent is truncated to ``i16``.

FP Classify
^^^^^^^^^^^

.. code-block:: c

   bool __builtin_amdgcn_class(double x, int mask);
   bool __builtin_amdgcn_classf(float x, int mask);
   bool __builtin_amdgcn_classh(__fp16 x, int mask);  // requires 16-bit-insts

Tests ``x`` against a bitmask of FP classes. Returns ``true`` if ``x`` matches
any of the selected classes. The ``mask`` bits are:

- Bit 0: Signaling NaN
- Bit 1: Quiet NaN
- Bit 2: Negative infinity
- Bit 3: Negative normal
- Bit 4: Negative denormal
- Bit 5: Negative zero
- Bit 6: Positive zero
- Bit 7: Positive denormal
- Bit 8: Positive normal
- Bit 9: Positive infinity

Median
^^^^^^

.. code-block:: c

   float  __builtin_amdgcn_fmed3f(float a, float b, float c);
   __fp16 __builtin_amdgcn_fmed3h(__fp16 a, __fp16 b, __fp16 c);  // requires gfx9-insts

Returns the median (middle value) of three floating-point numbers.

Cube Map Builtins
^^^^^^^^^^^^^^^^^

Require ``cube-insts``. All take three floats (x, y, z direction vector
components) and return a float.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Builtin
     - Description
   * - ``__builtin_amdgcn_cubeid(x, y, z)``
     - Returns the face ID (0-5) of the cube map.
   * - ``__builtin_amdgcn_cubesc(x, y, z)``
     - Returns the S coordinate for the cube face.
   * - ``__builtin_amdgcn_cubetc(x, y, z)``
     - Returns the T coordinate for the cube face.
   * - ``__builtin_amdgcn_cubema(x, y, z)``
     - Returns the major axis value.

Data Sharing Builtins
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Builtin
     - Description
   * - ``int __builtin_amdgcn_ds_swizzle(int data, _Constant int pattern)``
     - Performs a data-parallel swizzle within the wavefront according to the
       encoded ``pattern``.
   * - ``int __builtin_amdgcn_ds_permute(int addr, int data)``
     - Forward cross-lane permutation. Lane ``i`` gets the ``data`` value from
       the lane specified by ``addr / 4``.
   * - ``int __builtin_amdgcn_ds_bpermute(int addr, int data)``
     - Backward cross-lane permutation. Lane ``i`` reads from the lane
       specified by ``addr / 4``.
   * - ``int __builtin_amdgcn_ds_append(int __local *ptr)``
     - Atomically increments the value at ``ptr`` and returns the old value.
       The pointer must be in LDS (address space 3).
   * - ``int __builtin_amdgcn_ds_consume(int __local *ptr)``
     - Atomically decrements the value at ``ptr`` and returns the new value.

DS Float Atomics
^^^^^^^^^^^^^^^^

.. code-block:: c

   float __builtin_amdgcn_ds_faddf(float __local *ptr, float val, _Constant int ordering, _Constant int scope, _Constant bool isVolatile);
   float __builtin_amdgcn_ds_fminf(float __local *ptr, float val, _Constant int ordering, _Constant int scope, _Constant bool isVolatile);
   float __builtin_amdgcn_ds_fmaxf(float __local *ptr, float val, _Constant int ordering, _Constant int scope, _Constant bool isVolatile);

Perform atomic float add/min/max on LDS memory. The ``ordering`` and ``scope``
arguments are passed through but the operations are lowered to ``AtomicRMW``
instructions.

Lane Builtins
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Builtin
     - Description
   * - ``int __builtin_amdgcn_readfirstlane(int val)``
     - Returns the value of ``val`` from the first active lane.
   * - ``int __builtin_amdgcn_readlane(int val, int lane)``
     - Returns the value of ``val`` from the specified ``lane``.

Bit Manipulation
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Builtin
     - Description
   * - ``unsigned int __builtin_amdgcn_alignbit(unsigned int hi, unsigned int lo, unsigned int shift)``
     - Concatenates ``hi:lo`` as a 64-bit value and extracts 32 bits starting
       at bit ``shift``. Lowered to ``llvm.fshr``.
   * - ``unsigned int __builtin_amdgcn_alignbyte(unsigned int hi, unsigned int lo, unsigned int shift)``
     - Same as ``alignbit`` but ``shift`` is in bytes.
   * - ``unsigned int __builtin_amdgcn_ubfe(unsigned int base, unsigned int offset, unsigned int width)``
     - Unsigned bitfield extract from ``base`` starting at ``offset`` for
       ``width`` bits.
   * - ``unsigned int __builtin_amdgcn_sbfe(unsigned int base, unsigned int offset, unsigned int width)``
     - Signed bitfield extract.
   * - ``unsigned int __builtin_amdgcn_lerp(unsigned int a, unsigned int b, unsigned int c)``
     - Per-byte unsigned linear interpolation. Requires ``lerp-inst``.
   * - ``unsigned int __builtin_amdgcn_perm(unsigned int a, unsigned int b, unsigned int sel)``
     - Byte permutation. ``sel`` encodes which byte of the ``a:b`` pair to
       select for each byte of the result. Requires ``gfx8-insts``.

Conversion Builtins
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Builtin
     - Description
   * - ``half2 __builtin_amdgcn_cvt_pkrtz(float a, float b)``
     - Converts two f32 values to packed f16 with round-to-zero.
   * - ``short2 __builtin_amdgcn_cvt_pknorm_i16(float a, float b)``
     - Converts two f32 values to packed normalized i16. Requires
       ``cvt-pknorm-vop2-insts``.
   * - ``ushort2 __builtin_amdgcn_cvt_pknorm_u16(float a, float b)``
     - Converts two f32 values to packed normalized u16.
   * - ``short2 __builtin_amdgcn_cvt_pk_i16(int a, int b)``
     - Packs two i32 values into i16x2.
   * - ``ushort2 __builtin_amdgcn_cvt_pk_u16(unsigned int a, unsigned int b)``
     - Packs two u32 values into u16x2.
   * - ``unsigned int __builtin_amdgcn_cvt_pk_u8_f32(float val, unsigned int bytesel, unsigned int old)``
     - Converts ``val`` to u8 and inserts at byte ``bytesel`` in ``old``.
   * - ``float __builtin_amdgcn_cvt_off_f32_i4(int val)``
     - Converts a 4-bit integer offset to f32.

SAD (Sum of Absolute Differences)
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Builtin
     - Description
   * - ``unsigned int __builtin_amdgcn_msad_u8(unsigned int a, unsigned int b, unsigned int c)``
     - Masked sum of absolute differences of unsigned 8-bit values.
   * - ``unsigned int __builtin_amdgcn_sad_u8(unsigned int a, unsigned int b, unsigned int c)``
     - Sum of absolute differences of unsigned 8-bit values. Requires
       ``sad-insts``.
   * - ``unsigned int __builtin_amdgcn_sad_hi_u8(unsigned int a, unsigned int b, unsigned int c)``
     - SAD with result in high 16 bits. Requires ``sad-insts``.
   * - ``unsigned int __builtin_amdgcn_sad_u16(unsigned int a, unsigned int b, unsigned int c)``
     - SAD of unsigned 16-bit values. Requires ``sad-insts``.
   * - ``uint64_t __builtin_amdgcn_qsad_pk_u16_u8(uint64_t a, unsigned int b, uint64_t c)``
     - Quad SAD packed. Requires ``qsad-insts``.
   * - ``uint64_t __builtin_amdgcn_mqsad_pk_u16_u8(uint64_t a, unsigned int b, uint64_t c)``
     - Masked quad SAD packed.
   * - ``uint4 __builtin_amdgcn_mqsad_u32_u8(uint64_t a, unsigned int b, uint4 c)``
     - Masked quad SAD returning 4x u32.

Buffer Resource and Load/Store
==============================

make_buffer_rsrc
----------------

.. code-block:: c

   __amdgpu_buffer_rsrc_t __builtin_amdgcn_make_buffer_rsrc(void *base, short stride, int64_t num_records, int flags);

Constructs a buffer resource descriptor from the given fields:

- ``base``: Base pointer.
- ``stride``: Stride of structured buffer (0 for raw).
- ``num_records``: Number of records (bytes for raw buffers).
- ``flags``: SRD flags (DST_SEL, NUM_FORMAT, DATA_FORMAT, etc.).

Raw Buffer Load/Store
---------------------

These builtins load/store data through a buffer resource descriptor.

.. code-block:: c

   // Stores
   void __builtin_amdgcn_raw_buffer_store_b{8,16,32,64,96,128}(data, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, _Constant int cachepolicy);
   // Loads
   T __builtin_amdgcn_raw_buffer_load_b{8,16,32,64,96,128}(__amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, _Constant int cachepolicy);

Arguments:

- ``rsrc``: Buffer resource descriptor (128-bit SRD, typically SGPR).
- ``offset``: Byte offset (VGPR or immediate). Included in bounds checking and
  swizzling.
- ``soffset``: Scalar byte offset (SGPR or immediate). Excluded from bounds
  checking and swizzling.
- ``cachepolicy``: Immediate bitfield controlling cache behavior:

  - Pre-GFX12: bit 0 = GLC, bit 1 = SLC, bit 2 = DLC (gfx10/gfx11),
    bit 3 = SWZ, bit 4 = SCC (gfx90a).
  - GFX942: bit 0 = SC0, bit 1 = NT, bit 3 = SWZ, bit 4 = SC1.
  - GFX12+: bits [0:2] = TH, bits [3:4] = scope, bit 6 = SWZ.
  - All: bit 31 = volatile (stripped at lowering).

The data types for each width are: ``b8`` = ``unsigned char``,
``b16`` = ``unsigned short``, ``b32`` = ``unsigned int``,
``b64`` = ``uint2``, ``b96`` = ``uint3``, ``b128`` = ``uint4``.

Raw Ptr Buffer Atomics
----------------------

.. code-block:: c

   int   __builtin_amdgcn_raw_ptr_buffer_atomic_add_i32(int val, __amdgpu_buffer_rsrc_t rsrc, int offset, int soffset, _Constant int cachepolicy);
   float __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32(...);     // requires atomic-fadd-rtn-insts
   half2 __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16(...);   // requires atomic-buffer-global-pk-add-f16-insts
   float __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f32(...);     // requires atomic-fmin-fmax-global-f32
   float __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32(...);     // requires atomic-fmin-fmax-global-f32
   double __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64(...);    // requires atomic-fmin-fmax-global-f64
   double __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64(...);    // requires atomic-fmin-fmax-global-f64

Same argument layout as raw buffer loads: ``(val, rsrc, offset, soffset, cachepolicy)``.

Buffer-to-LDS Load Builtins
---------------------------

.. code-block:: c

   // Raw pointer variants
   void __builtin_amdgcn_raw_ptr_buffer_load_lds(__amdgpu_buffer_rsrc_t rsrc, void __local *lds_ptr, _Constant unsigned int size, int offset, int soffset, _Constant int aux, _Constant int cachepolicy);
   void __builtin_amdgcn_raw_ptr_buffer_load_async_lds(...);  // same args
   // Struct pointer variants (extra vindex arg)
   void __builtin_amdgcn_struct_ptr_buffer_load_lds(__amdgpu_buffer_rsrc_t rsrc, void __local *lds_ptr, _Constant unsigned int size, int vindex, int offset, int soffset, _Constant int aux, _Constant int cachepolicy);
   void __builtin_amdgcn_struct_ptr_buffer_load_async_lds(...);

All require ``vmem-to-lds-load-insts``.

**Restriction on ``size``**: Must be a compile-time constant. Valid values are
**1, 2, 4** on most targets, or **1, 2, 4, 12, 16** on targets with
``gfx950-insts``.

Async / LDS Load Builtins
=========================

.. code-block:: c

   void __builtin_amdgcn_load_to_lds(void *src, void __local *dst, _Constant unsigned int size, _Constant int offset, _Constant unsigned int aux);
   void __builtin_amdgcn_load_async_to_lds(void *src, void __local *dst, _Constant unsigned int size, _Constant int offset, _Constant unsigned int aux);
   void __builtin_amdgcn_global_load_lds(void __global *src, void __local *dst, _Constant unsigned int size, _Constant int offset, _Constant unsigned int aux);
   void __builtin_amdgcn_global_load_async_lds(void __global *src, void __local *dst, _Constant unsigned int size, _Constant int offset, _Constant unsigned int aux);

All require ``vmem-to-lds-load-insts``.

- ``src``: Source pointer (flat or global address space).
- ``dst``: Destination pointer in LDS (address space 3).
- ``size``: Number of bytes to load. **Must be constant**. Valid: 1, 2, 4
  (or 1, 2, 4, 12, 16 with ``gfx950-insts``).
- ``offset``: Byte offset applied to ``dst``.
- ``aux``: Auxiliary cache policy.

Async Mark Builtins
-------------------

.. code-block:: c

   void __builtin_amdgcn_asyncmark();
   void __builtin_amdgcn_wait_asyncmark(_Constant unsigned short count);

Mark and wait for asynchronous operations. Require ``vmem-to-lds-load-insts``.

Ballot and Wave Builtins
========================

Ballot
------

.. code-block:: c

   uint32_t __builtin_amdgcn_ballot_w32(bool pred);  // requires wavefrontsize32
   uint64_t __builtin_amdgcn_ballot_w64(bool pred);

Returns a bitmask of active lanes where ``pred`` is ``true``.

Inverse Ballot
--------------

.. code-block:: c

   bool __builtin_amdgcn_inverse_ballot_w32(uint32_t mask);  // requires wavefrontsize32
   bool __builtin_amdgcn_inverse_ballot_w64(uint64_t mask);  // requires wavefrontsize64

Returns ``true`` for the current lane if the corresponding bit in ``mask`` is
set.

Deprecated Compare Builtins
---------------------------

These are deprecated in favor of ``ballot``:

.. code-block:: c

   uint64_t __builtin_amdgcn_uicmp(unsigned int a, unsigned int b, _Constant int cmp);
   uint64_t __builtin_amdgcn_uicmpl(uint64_t a, uint64_t b, _Constant int cmp);
   uint64_t __builtin_amdgcn_sicmp(int a, int b, _Constant int cmp);
   uint64_t __builtin_amdgcn_sicmpl(int64_t a, int64_t b, _Constant int cmp);
   uint64_t __builtin_amdgcn_fcmp(double a, double b, _Constant int cmp);
   uint64_t __builtin_amdgcn_fcmpf(float a, float b, _Constant int cmp);

The ``cmp`` constant selects the comparison predicate (uses LLVM ICmp/FCmp
predicate encoding).

Wave Reduction Builtins
-----------------------

Perform a reduction across all active lanes in the wavefront. All take the
form:

.. code-block:: c

   T __builtin_amdgcn_wave_reduce_<op>_<type>(T val, _Constant int32_t strategy);

Available operations and types:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Operation
     - Types
   * - ``add``, ``sub``
     - ``u32``, ``u64``
   * - ``min``, ``max``
     - ``i32``, ``u32``, ``i64``, ``u64``
   * - ``and``, ``or``, ``xor``
     - ``b32``, ``b64``
   * - ``fadd``, ``fsub``, ``fmin``, ``fmax``
     - ``f32``, ``f64``

The ``strategy`` argument is an implementation strategy hint.

Flat Addressing Builtins
========================

.. code-block:: c

   bool __builtin_amdgcn_is_shared(const void *ptr);
   bool __builtin_amdgcn_is_private(const void *ptr);

Tests whether the given flat pointer actually points to shared (LDS) or private
(scratch) memory.

GWS (Global Wave Sync) Builtins
===============================

All require ``gws``.

.. code-block:: c

   void __builtin_amdgcn_ds_gws_init(unsigned int val, unsigned int id);
   void __builtin_amdgcn_ds_gws_barrier(unsigned int val, unsigned int id);
   void __builtin_amdgcn_ds_gws_sema_v(unsigned int id);
   void __builtin_amdgcn_ds_gws_sema_br(unsigned int val, unsigned int id);
   void __builtin_amdgcn_ds_gws_sema_p(unsigned int id);
   void __builtin_amdgcn_ds_gws_sema_release_all(unsigned int id);  // requires ci-insts

- ``val``: Value or count for the GWS operation.
- ``id``: GWS resource ID.

CI+ Cache Control
-----------------

.. code-block:: c

   void __builtin_amdgcn_s_dcache_inv_vol();     // requires ci-insts
   void __builtin_amdgcn_buffer_wbinvl1_vol();    // requires ci-insts

Invalidate volatile data cache and buffer L1 cache.

Interpolation Builtins
======================

.. code-block:: c

   float  __builtin_amdgcn_interp_p1(float i, unsigned int attr_chan, unsigned int attr, unsigned int m0);
   float  __builtin_amdgcn_interp_p2(float p1, float j, unsigned int attr_chan, unsigned int attr, unsigned int m0);
   float  __builtin_amdgcn_interp_mov(unsigned int param, unsigned int attr_chan, unsigned int attr, unsigned int m0);
   float  __builtin_amdgcn_interp_p1_f16(float i, unsigned int attr_chan, unsigned int attr, bool high, unsigned int m0);
   __fp16 __builtin_amdgcn_interp_p2_f16(float p1, float j, unsigned int attr_chan, unsigned int attr, bool high, unsigned int m0);

- ``i``, ``j``: Interpolation coordinates (barycentric).
- ``attr_chan``: Attribute channel (0-3 for x/y/z/w).
- ``attr``: Attribute index.
- ``param``: For ``interp_mov``, selects the parameter to copy (P0, P10, P20).
- ``high``: For f16 variants, selects the high or low half of the attribute.
- ``m0``: The M0 register value (parameter base offset).

DPP (Data-Parallel Primitives) Builtins
=======================================

Require ``dpp`` feature.

``__builtin_amdgcn_mov_dpp``
----------------------------

.. code-block:: c

   int __builtin_amdgcn_mov_dpp(int src, _Constant int dpp_ctrl, _Constant int row_mask, _Constant int bank_mask, _Constant bool bound_ctrl);

Performs a DPP data movement with the given control pattern. The data arguments
must be arithmetic (non-complex) types.

- ``src``: Source data.
- ``dpp_ctrl``: DPP control word encoding the permutation pattern.
- ``row_mask``: 4-bit mask selecting which rows participate.
- ``bank_mask``: 4-bit mask selecting which banks participate.
- ``bound_ctrl``: If true, out-of-bounds lanes get zero instead of their own
  value.

**Lowering note**: A poison value is prepended as the "old" operand. For types
smaller than 32 bits, the value is zero-extended to i32 for the intrinsic, then
truncated back.

``__builtin_amdgcn_update_dpp``
-------------------------------

.. code-block:: c

   int __builtin_amdgcn_update_dpp(int old, int src, _Constant int dpp_ctrl, _Constant int row_mask, _Constant int bank_mask, _Constant bool bound_ctrl);

Like ``mov_dpp`` but with an explicit ``old`` value for out-of-bounds or
masked-off lanes. Sema validates that ``old`` and ``src`` have the same type
(or compatible signed/unsigned pair).

``__builtin_amdgcn_mov_dpp8``
-----------------------------

.. code-block:: c

   unsigned int __builtin_amdgcn_mov_dpp8(unsigned int src, _Constant unsigned int sel);

Requires ``gfx10-insts``. Performs a DPP8 permutation with ``sel`` encoding
8 3-bit lane selectors.

Permute / Lane Builtins
=======================

GFX10+ Permute Lane
-------------------

.. code-block:: c

   unsigned int __builtin_amdgcn_permlane16(unsigned int old, unsigned int src, unsigned int src1, unsigned int src2, _Constant bool fi, _Constant bool bc);
   unsigned int __builtin_amdgcn_permlanex16(unsigned int old, unsigned int src, unsigned int src1, unsigned int src2, _Constant bool fi, _Constant bool bc);

Requires ``gfx10-insts``.

- ``old``: Value for inactive lanes.
- ``src``: Source value.
- ``src1``, ``src2``: Lane index selects (SGPR).
- ``fi``: Fetch inactive -- if true, fetch from inactive lanes.
- ``bc``: Bound control -- if true, out-of-bound lanes get their own value.

GFX12+ Variable Permute Lane
----------------------------

.. code-block:: c

   unsigned int __builtin_amdgcn_permlane16_var(unsigned int old, unsigned int src, unsigned int lane_sel, _Constant bool fi, _Constant bool bc);
   unsigned int __builtin_amdgcn_permlanex16_var(unsigned int old, unsigned int src, unsigned int lane_sel, _Constant bool fi, _Constant bool bc);

Requires ``gfx12-insts``. Like the above but ``lane_sel`` is a VGPR.

GFX11+ permlane64
-----------------

.. code-block:: c

   unsigned int __builtin_amdgcn_permlane64(unsigned int src);  // requires gfx11-insts

Swaps data between lane ``i`` and lane ``i + 32`` (i.e., between the two
halves of a wave64). This is a no-op in wave32.

GFX950 Permute Lane Swap
------------------------

.. code-block:: c

   uint2 __builtin_amdgcn_permlane16_swap(unsigned int old, unsigned int src, _Constant bool fi, _Constant bool bc);
   uint2 __builtin_amdgcn_permlane32_swap(unsigned int old, unsigned int src, _Constant bool fi, _Constant bool bc);

Requires ``permlane16-swap`` / ``permlane32-swap``. Returns a 2-element vector:
element 0 is the current lane result, element 1 is the swapped lane result.

**Lowering note**: The intrinsic returns ``{i32, i32}``; the builtin packs
this into ``<2 x i32>``.

GFX1250 Permute Lane Builtins
-----------------------------

Require ``gfx1250-insts,wavefrontsize32``.

.. code-block:: c

   int __builtin_amdgcn_permlane_bcast(int old, int src, int lane_sel);
   int __builtin_amdgcn_permlane_up(int old, int src, int lane_sel);
   int __builtin_amdgcn_permlane_down(int old, int src, int lane_sel);
   int __builtin_amdgcn_permlane_xor(int old, int src, int lane_sel);
   int __builtin_amdgcn_permlane_idx_gen(int src, int lane_sel);

Deep Learning / Dot Product Builtins
====================================

These builtins perform dot product accumulate operations used in deep learning
workloads.

Float Dot Product
-----------------

.. code-block:: c

   float  __builtin_amdgcn_fdot2(half2 a, half2 b, float c, _Constant bool clamp);          // dot10-insts
   __fp16 __builtin_amdgcn_fdot2_f16_f16(half2 a, half2 b, __fp16 c);                       // dot9-insts
   short  __builtin_amdgcn_fdot2_bf16_bf16(short2 a, short2 b, short c);                     // dot9-insts
   float  __builtin_amdgcn_fdot2_f32_bf16(short2 a, short2 b, float c, _Constant bool clamp); // dot12-insts
   float  __builtin_amdgcn_fdot2c_f32_bf16(bf16x2 a, bf16x2 b, float c, _Constant bool clamp); // dot13-insts

Computes ``dot(a, b) + c``. The ``clamp`` flag, when present and true, clamps
the result to [0.0, 1.0].

Integer Dot Product
-------------------

.. code-block:: c

   int          __builtin_amdgcn_sdot2(short2 a, short2 b, int c, _Constant bool clamp);    // dot2-insts
   unsigned int __builtin_amdgcn_udot2(ushort2 a, ushort2 b, unsigned int c, _Constant bool clamp); // dot2-insts
   int          __builtin_amdgcn_sdot4(int a, int b, int c, _Constant bool clamp);           // dot1-insts
   unsigned int __builtin_amdgcn_udot4(unsigned int a, unsigned int b, unsigned int c, _Constant bool clamp); // dot7-insts
   int          __builtin_amdgcn_sdot8(int a, int b, int c, _Constant bool clamp);           // dot1-insts
   unsigned int __builtin_amdgcn_udot8(unsigned int a, unsigned int b, unsigned int c, _Constant bool clamp); // dot7-insts

For ``sdot4``/``udot4``, each 32-bit argument is treated as four packed 8-bit
integers. For ``sdot8``/``udot8``, they are eight packed 4-bit integers. The
``clamp`` flag saturates on overflow.

Mixed Sign Dot Product
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   int __builtin_amdgcn_sudot4(_Constant bool a_sign, int a, _Constant bool b_sign, int b, int c, _Constant bool clamp);  // dot8-insts
   int __builtin_amdgcn_sudot8(_Constant bool a_sign, int a, _Constant bool b_sign, int b, int c, _Constant bool clamp);  // dot8-insts

- ``a_sign``, ``b_sign``: If true, treat the corresponding operand as signed.
  If false, treat as unsigned.

FP8/BF8 Dot Product
^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   float __builtin_amdgcn_dot4_f32_fp8_fp8(unsigned int a, unsigned int b, float c);  // dot11-insts
   float __builtin_amdgcn_dot4_f32_fp8_bf8(unsigned int a, unsigned int b, float c);
   float __builtin_amdgcn_dot4_f32_bf8_fp8(unsigned int a, unsigned int b, float c);
   float __builtin_amdgcn_dot4_f32_bf8_bf8(unsigned int a, unsigned int b, float c);

Each 32-bit input encodes 4 packed 8-bit floats (FP8 or BF8).

Raytracing / BVH Builtins
=========================

GFX10+ BVH Intersect Ray
------------------------

.. code-block:: c

   uint4 __builtin_amdgcn_image_bvh_intersect_ray(uint node, float ray_extent, float4 ray_origin, float4 ray_dir, float4 ray_inv_dir, uint4 texture_desc);
   uint4 __builtin_amdgcn_image_bvh_intersect_ray_h(uint node, float ray_extent, float4 ray_origin, half4 ray_dir, half4 ray_inv_dir, uint4 texture_desc);
   uint4 __builtin_amdgcn_image_bvh_intersect_ray_l(uint64_t node, float ray_extent, float4 ray_origin, float4 ray_dir, float4 ray_inv_dir, uint4 texture_desc);
   uint4 __builtin_amdgcn_image_bvh_intersect_ray_lh(uint64_t node, float ray_extent, float4 ray_origin, half4 ray_dir, half4 ray_inv_dir, uint4 texture_desc);

Requires ``gfx10-insts``. Naming convention: ``l`` = 64-bit node pointer;
``h`` = half-precision direction vectors.

- ``node``: BVH node pointer.
- ``ray_extent``: Maximum ray distance.
- ``ray_origin``: Ray origin (only xyz used).
- ``ray_dir``: Ray direction (only xyz used).
- ``ray_inv_dir``: Inverse ray direction (only xyz used).
- ``texture_desc``: Image descriptor for the BVH.

**Lowering note**: The builtin takes ``float4``/``half4`` vectors but only
the first 3 components are used. The lowering uses a shuffle to extract
xyz (vec3) before passing to the intrinsic.

GFX12+ BVH Builtins
-------------------

.. code-block:: c

   uint10 __builtin_amdgcn_image_bvh8_intersect_ray(uint64_t node, float ray_extent, unsigned char ray_mask, float3 ray_origin, float3 ray_dir, unsigned int offset, uint4 texture_desc, float3 *ray_origin_out, float3 *ray_dir_out);
   uint10 __builtin_amdgcn_image_bvh_dual_intersect_ray(uint64_t node, float ray_extent, unsigned char ray_mask, float3 ray_origin, float3 ray_dir, uint2 offsets, uint4 texture_desc, float3 *ray_origin_out, float3 *ray_dir_out);

Requires ``gfx12-insts``. The intrinsic returns ``{vdata, ray_origin, ray_dir}``
as a struct. The builtin returns ``vdata`` and writes the updated ray origin
and direction through the pointer arguments.

DS BVH Stack Builtins
---------------------

.. code-block:: c

   uint2 __builtin_amdgcn_ds_bvh_stack_rtn(unsigned int addr, unsigned int data, uint4 input, _Constant int controls);       // gfx11-insts
   uint2 __builtin_amdgcn_ds_bvh_stack_push4_pop1_rtn(unsigned int addr, unsigned int data, uint4 input, _Constant int controls); // gfx11-insts
   uint2 __builtin_amdgcn_ds_bvh_stack_push8_pop1_rtn(unsigned int addr, unsigned int data, uint8 input, _Constant int controls); // gfx12-insts
   uint64_t2 __builtin_amdgcn_ds_bvh_stack_push8_pop2_rtn(unsigned int addr, unsigned int data, uint8 input, _Constant int controls); // gfx12-insts

The ``push8_pop2`` variant returns ``<2 x i64>``; the intrinsic's second
``i32`` return value is zero-extended to ``i64``.

MFMA (Matrix Fused Multiply-Add) Builtins
=========================================

MFMA builtins perform matrix multiply-accumulate operations. They require
``mai-insts``, ``fp8-insts``, or ``gfx950-insts`` depending on the variant.

General Signature
-----------------

.. code-block:: c

   vecN __builtin_amdgcn_mfma_<outtype>_MxNxK<intype>(A a, B b, vecN c, _Constant int cbsz, _Constant int abid, _Constant int blgp);

- ``a``, ``b``: Input matrix fragments.
- ``c``: Accumulator matrix fragment.
- ``cbsz``: Control broadcast size (broadcast mode for A operand).
- ``abid``: Accumulator broadcast ID (selects which accumulator portion
  to use).
- ``blgp``: B-matrix lane group pattern.

All three control arguments (``cbsz``, ``abid``, ``blgp``) must be compile-time
constants.

The matrix dimensions (M, N, K) and input/output types are encoded in the
builtin name. For example, ``mfma_f32_32x32x1f32`` computes a 32x32x1 matrix
multiply with f32 inputs and f32 accumulator.

SMFMAC (Sparse Matrix FMA)
--------------------------

.. code-block:: c

   vecN __builtin_amdgcn_smfmac_<outtype>_MxNxK<intype>(A a, B b, vecN c, int idx, _Constant int cbsz, _Constant int abid);

- ``a``, ``b``: Sparse input matrix fragments.
- ``c``: Accumulator.
- ``idx``: Sparse index (non-constant, VGPR).
- ``cbsz``, ``abid``: Constant control parameters.

GFX950 MFMA Scale
-----------------

.. code-block:: c

   vec4  __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(int8 a, int8 b, vec4 c, _Constant int cbsz, _Constant int abid, _Constant int blgp, int a_scale, _Constant int a_scale_fmt, int b_scale);
   vec16 __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(int8 a, int8 b, vec16 c, _Constant int cbsz, _Constant int abid, _Constant int blgp, int a_scale, _Constant int a_scale_fmt, int b_scale);

Requires ``gfx950-insts``. Adds per-operand scaling factors:

- ``a_scale``, ``b_scale``: Scale factors for A and B matrices.
- ``a_scale_fmt``: Scale format selector (constant).

FP8 Conversion Builtins
=======================

These builtins convert between FP8/BF8 formats and standard float types.
Require ``fp8-conversion-insts`` or related features.

.. code-block:: c

   float __builtin_amdgcn_cvt_f32_bf8(int src, _Constant int byte_sel);  // fp8-conversion-insts
   float __builtin_amdgcn_cvt_f32_fp8(int src, _Constant int byte_sel);
   float __builtin_amdgcn_cvt_f32_fp8_e5m3(int src, _Constant int byte_sel);  // fp8e5m3-insts

- ``src``: Packed 4x FP8/BF8 values in an ``int``.
- ``byte_sel``: Selects which byte (0-3) to convert.

.. code-block:: c

   float2 __builtin_amdgcn_cvt_pk_f32_bf8(int src, _Constant bool word_sel);
   float2 __builtin_amdgcn_cvt_pk_f32_fp8(int src, _Constant bool word_sel);

- ``word_sel``: If false, converts bytes [0:1]; if true, converts bytes [2:3].

.. code-block:: c

   int __builtin_amdgcn_cvt_pk_bf8_f32(float a, float b, int old, _Constant bool word_sel);
   int __builtin_amdgcn_cvt_pk_fp8_f32(float a, float b, int old, _Constant bool word_sel);

- ``a``, ``b``: Two f32 values to pack into FP8.
- ``old``: Previous packed value (the unselected half is preserved).
- ``word_sel``: Selects which half of the result to write.

.. code-block:: c

   int __builtin_amdgcn_cvt_sr_bf8_f32(float val, int rng, int old, _Constant int byte_sel);
   int __builtin_amdgcn_cvt_sr_fp8_f32(float val, int rng, int old, _Constant int byte_sel);

- ``val``: Input float.
- ``rng``: Random bits for stochastic rounding.
- ``old``: Previous packed result (unselected bytes preserved).
- ``byte_sel``: Which byte position (0-3) to write.

WMMA (Wave Matrix Multiply-Accumulate) Builtins
===============================================

WMMA builtins perform cooperative matrix multiply operations across a wavefront.
The ``_w32`` suffix indicates wave32 requirement, ``_w64`` for wave64.

GFX11 WMMA
----------

Requires ``gfx11-insts`` and the appropriate wavefront size.

**Float/Half WMMA** (16x16x16):

.. code-block:: c

   // Wave32 variants
   float8  __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(half16 a, half16 b, float8 c);
   float8  __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(short16 a, short16 b, float8 c);
   half16  __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(half16 a, half16 b, half16 c, _Constant bool opsel);
   short16 __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(short16 a, short16 b, short16 c, _Constant bool opsel);

- ``a``, ``b``: Input matrix fragments.
- ``c``: Accumulator.
- ``opsel``: Operation select -- controls which half of the accumulator to
  write (for f16/bf16 output variants).

The ``_tied`` variants (e.g., ``wmma_f16_16x16x16_f16_tied_w32``) have the
same signature but indicate the output is tied to the input accumulator
register.

**Integer WMMA**:

.. code-block:: c

   int8 __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(_Constant bool a_sign, int4 a, _Constant bool b_sign, int4 b, int8 c, _Constant bool clamp);
   int8 __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(_Constant bool a_sign, int2 a, _Constant bool b_sign, int2 b, int8 c, _Constant bool clamp);

- ``a_sign``, ``b_sign``: If true, treat operand as signed.
- ``clamp``: If true, saturate the result.

Wave64 variants have the same argument pattern but smaller vector sizes
(e.g., ``float4`` instead of ``float8``).

GFX12 WMMA
----------

Requires ``gfx12-insts``. Distinguished by ``_gfx12`` suffix. Smaller input
vectors (no A/B replication required). Adds FP8/BF8 variants:

.. code-block:: c

   float8 __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(int2 a, int2 b, float8 c);
   // ... similar for fp8_bf8, bf8_fp8, bf8_bf8

GFX1250 WMMA
------------

Requires ``gfx1250-insts,wavefrontsize32``. Extended variants with additional
control:

.. code-block:: c

   float8 __builtin_amdgcn_wmma_f32_16x16x4_f32(_Constant bool a_neg, float2 a, _Constant bool b_neg, float2 b, _Constant short matrix_fmt, float8 c, _Constant bool neg_lo, _Constant bool neg_hi);

- ``a_neg``, ``b_neg``: Negate input operands.
- ``matrix_fmt``: Matrix format selector.
- ``neg_lo``, ``neg_hi``: Negate low/high halves of accumulator.

The ``_iu8`` variant accepts an optional clamp argument (7 or 8 args):

.. code-block:: c

   int8 __builtin_amdgcn_wmma_i32_16x16x64_iu8(_Constant bool a_sign, int8 a, _Constant bool b_sign, int8 b, int8 c, _Constant bool neg_lo, _Constant bool neg_hi[, _Constant bool clamp]);

**Restriction**: When the optional ``clamp`` argument is present, it must be
a compile-time constant boolean.

SWMMAC (Sparse Wave Matrix Multiply-Accumulate)
===============================================

GFX12 SWMMAC
------------

Requires ``gfx12-insts``. The ``index`` argument provides sparsity
metadata.

.. code-block:: c

   float8  __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(half8 a, half16 b, float8 c, int index);
   float8  __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(short8 a, short16 b, float8 c, int index);
   half8   __builtin_amdgcn_swmmac_f16_16x16x32_f16_w32(half8 a, half16 b, half8 c, int index);
   short8  __builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32(short8 a, short16 b, short8 c, int index);

**Integer SWMMAC**:

.. code-block:: c

   int8 __builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32(_Constant bool a_sign, int2 a, _Constant bool b_sign, int4 b, int8 c, int index, _Constant bool clamp);
   int8 __builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32(_Constant bool a_sign, int a, _Constant bool b_sign, int2 b, int8 c, int index, _Constant bool clamp);

**FP8/BF8 SWMMAC**:

.. code-block:: c

   float8 __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(int2 a, int4 b, float8 c, int index);
   // ... similar for fp8_bf8, bf8_fp8, bf8_bf8

GFX1250 SWMMAC
--------------

Requires ``gfx1250-insts,wavefrontsize32``. Extended variants with negation
control and additional ``neg_lo``, ``neg_hi`` flags.

The ``_iu8`` variant also accepts an optional clamp argument:

.. code-block:: c

   int8 __builtin_amdgcn_swmmac_i32_16x16x128_iu8(_Constant bool a_sign, int8 a, _Constant bool b_sign, int16 b, int8 c, int index, _Constant bool neg_lo, _Constant bool neg_hi[, _Constant bool clamp]);

Atomic Builtins
===============

Global Atomics
--------------

.. code-block:: c

   double __builtin_amdgcn_global_atomic_fadd_f64(double __global *ptr, double val);   // gfx90a-insts
   float  __builtin_amdgcn_global_atomic_fadd_f32(float __global *ptr, float val);     // atomic-fadd-rtn-insts
   half2  __builtin_amdgcn_global_atomic_fadd_v2f16(half2 __global *ptr, half2 val);   // atomic-buffer-global-pk-add-f16-insts
   double __builtin_amdgcn_global_atomic_fmin_f64(double __global *ptr, double val);   // gfx90a-insts
   double __builtin_amdgcn_global_atomic_fmax_f64(double __global *ptr, double val);   // gfx90a-insts
   short2 __builtin_amdgcn_global_atomic_fadd_v2bf16(short2 __global *ptr, short2 val); // atomic-global-pk-add-bf16-inst

Flat Atomics
------------

.. code-block:: c

   double __builtin_amdgcn_flat_atomic_fadd_f64(double *ptr, double val);    // gfx90a-insts
   double __builtin_amdgcn_flat_atomic_fmin_f64(double *ptr, double val);    // gfx90a-insts
   double __builtin_amdgcn_flat_atomic_fmax_f64(double *ptr, double val);    // gfx90a-insts
   float  __builtin_amdgcn_flat_atomic_fadd_f32(float *ptr, float val);     // gfx940-insts
   half2  __builtin_amdgcn_flat_atomic_fadd_v2f16(half2 *ptr, half2 val);   // atomic-flat-pk-add-16-insts
   short2 __builtin_amdgcn_flat_atomic_fadd_v2bf16(short2 *ptr, short2 val); // atomic-flat-pk-add-16-insts

DS (LDS) Atomics
----------------

.. code-block:: c

   double __builtin_amdgcn_ds_atomic_fadd_f64(double __local *ptr, double val);     // gfx90a-insts
   float  __builtin_amdgcn_ds_atomic_fadd_f32(float __local *ptr, float val);      // gfx8-insts
   short2 __builtin_amdgcn_ds_atomic_fadd_v2bf16(short2 __local *ptr, short2 val); // atomic-ds-pk-add-16-insts
   half2  __builtin_amdgcn_ds_atomic_fadd_v2f16(half2 __local *ptr, half2 val);    // atomic-ds-pk-add-16-insts

**Lowering note**: All global, flat, and DS atomic builtins are lowered to
``AtomicRMW`` IR instructions (not intrinsic calls).

Atomic Inc/Dec
--------------

.. code-block:: c

   uint32_t __builtin_amdgcn_atomic_inc32(volatile uint32_t *ptr, uint32_t val, unsigned int ordering, const char *scope);
   uint64_t __builtin_amdgcn_atomic_inc64(volatile uint64_t *ptr, uint64_t val, unsigned int ordering, const char *scope);
   uint32_t __builtin_amdgcn_atomic_dec32(volatile uint32_t *ptr, uint32_t val, unsigned int ordering, const char *scope);
   uint64_t __builtin_amdgcn_atomic_dec64(volatile uint64_t *ptr, uint64_t val, unsigned int ordering, const char *scope);

- ``ptr``: Pointer to the atomic variable.
- ``val``: Comparand value.
- ``ordering``: C11 atomic memory ordering constant.
- ``scope``: Synchronization scope string (e.g., ``"workgroup"``).

**Restrictions**: ``ordering`` must be a valid C11/C++11 atomic ordering.
``scope`` must be a string literal that is a valid synchronization scope.

Fence
=====

.. code-block:: c

   void __builtin_amdgcn_fence(unsigned int ordering, const char *scope, ...);

Inserts a memory fence with the given ordering and synchronization scope.

- ``ordering``: C11 atomic ordering. ``relaxed`` and ``consume`` are not
  allowed for fences.
- ``scope``: Synchronization scope string literal (e.g., ``"workgroup"``,
  ``"agent"``, ``"system"``).
- Optional variadic arguments specify address spaces affected.

**Lowering note**: Lowered to an LLVM ``fence`` instruction, not an intrinsic
call.

Scheduling / Wait Builtins
==========================

Scheduling Barriers
-------------------

``__builtin_amdgcn_sched_barrier``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   void __builtin_amdgcn_sched_barrier(_Constant int mask);

Controls which instruction types may be reordered across this point:

- ``0x0000``: No instructions may cross.
- ``0x0001``: ALU (non-memory, non-side-effect).
- ``0x0002``: VALU.
- ``0x0004``: SALU.
- ``0x0008``: MFMA/WMMA.
- ``0x0010``: All VMEM.
- ``0x0020``: VMEM reads.
- ``0x0040``: VMEM writes.
- ``0x0080``: All DS.
- ``0x0100``: DS reads.
- ``0x0200``: DS writes.

``__builtin_amdgcn_sched_group_barrier``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   void __builtin_amdgcn_sched_group_barrier(_Constant int mask, _Constant int count, _Constant int group_id);

- ``mask``: Same encoding as ``sched_barrier``.
- ``count``: Number of instructions in this scheduling group.
- ``group_id``: Identifier for synchronizing with other sched_group_barriers.

``__builtin_amdgcn_iglp_opt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c

   void __builtin_amdgcn_iglp_opt(_Constant int strategy);

Scheduler optimization hint. ``strategy = 0`` enables small GEMM optimization.

Wait Events
-----------

.. code-block:: c

   void __builtin_amdgcn_s_wait_event_export_ready();  // gfx11-insts
   void __builtin_amdgcn_s_wait_event(_Constant short mask);  // gfx11-insts

**``s_wait_event`` restrictions**: The ``mask`` argument has target-specific
valid bits:

- GFX11: Bit 0 should be 0; setting bit 0 generates a warning.
- GFX12+: Bit 1 should be 0; clearing bit 1 generates a warning.
- The suggested value for ``export_ready`` is 2 on both GFX11 and GFX12+.
- Values > 2 generate a warning.

Sleep
-----

.. code-block:: c

   void __builtin_amdgcn_s_sleep(_Constant int duration);
   void __builtin_amdgcn_s_sleep_var(unsigned int duration);   // gfx12-insts

The ``_var`` variant allows a non-constant duration.

GFX11+ Message Return
---------------------

.. code-block:: c

   unsigned int __builtin_amdgcn_s_sendmsg_rtn(_Constant unsigned int msg);  // gfx11-insts
   uint64_t __builtin_amdgcn_s_sendmsg_rtnl(_Constant unsigned int msg);    // gfx11-insts

Send a message and return a value.

Special Builtins
================

.. code-block:: c

   uint64_t     __builtin_amdgcn_read_exec();
   unsigned int __builtin_amdgcn_read_exec_lo();
   unsigned int __builtin_amdgcn_read_exec_hi();

Read the current EXEC mask. These are **not** lowered to dedicated intrinsics
but synthesized via ``ballot(true)``. ``read_exec_lo`` returns the lower 32
bits; ``read_exec_hi`` shifts right by 32 and truncates.

.. code-block:: c

   void __builtin_amdgcn_endpgm();

Terminates the shader/kernel. Marked ``NoReturn``.

.. code-block:: c

   uint64_t __builtin_amdgcn_get_fpenv();
   void     __builtin_amdgcn_set_fpenv(uint64_t env);

Get/set the floating-point environment (MODE register). Lowered to
``llvm.get.fpenv`` / ``llvm.set.fpenv``.

VI+ Miscellaneous
-----------------

.. code-block:: c

   uint64_t __builtin_amdgcn_s_memrealtime();  // requires s-memrealtime
   void __builtin_amdgcn_s_dcache_wb();        // requires gfx8-insts

GFX12+ Barrier Builtins
=======================

Require ``gfx12-insts``.

.. code-block:: c

   void __builtin_amdgcn_s_barrier_signal(_Constant int bar_id);
   void __builtin_amdgcn_s_barrier_signal_var(void *bar, int member_count);
   bool __builtin_amdgcn_s_barrier_signal_isfirst(_Constant int bar_id);
   void __builtin_amdgcn_s_barrier_wait(_Constant short signal_count);
   void __builtin_amdgcn_s_barrier_init(void *bar, int member_count);
   void __builtin_amdgcn_s_barrier_join(void *bar);
   void __builtin_amdgcn_s_barrier_leave(_Constant short flags);
   unsigned int __builtin_amdgcn_s_get_barrier_state(int bar_id);
   unsigned int __builtin_amdgcn_s_get_named_barrier_state(void *bar);

- ``bar_id``: Barrier ID (constant for signal/wait, variable for get_state).
- ``bar``: Pointer to named barrier object.
- ``member_count``: Number of wavefronts participating.
- ``signal_count``: Number of signals to wait for.

Prefetch Builtins
-----------------

.. code-block:: c

   void __builtin_amdgcn_s_prefetch_data(const void *ptr, unsigned int length);  // gfx12-insts
   void __builtin_amdgcn_s_buffer_prefetch_data(__amdgpu_buffer_rsrc_t rsrc, _Constant int offset, unsigned int length); // gfx12-insts

GFX12+ Transport Loads
----------------------

.. code-block:: c

   int2   __builtin_amdgcn_global_load_tr_b64_v2i32(int2 __global *ptr);    // gfx12-insts,wavefrontsize32
   short8 __builtin_amdgcn_global_load_tr_b128_v8i16(short8 __global *ptr); // gfx12-insts,wavefrontsize32
   // ... also half8, bf16x8 variants
   int    __builtin_amdgcn_global_load_tr_b64_i32(int __global *ptr);       // gfx12-insts,wavefrontsize64
   short4 __builtin_amdgcn_global_load_tr_b128_v4i16(short4 __global *ptr); // gfx12-insts,wavefrontsize64

Transport loads for matrix operands. Pointer must be in global address space.

.. code-block:: c

   int __builtin_amdgcn_ds_bpermute_fi_b32(int addr, int data);  // gfx12-insts

DS backward permutation with fetch-inactive.

GFX12+ Bitwise Ternary Operation
--------------------------------

.. code-block:: c

   int   __builtin_amdgcn_bitop3_b32(int a, int b, int c, _Constant unsigned int lut);  // bitop3-insts
   short __builtin_amdgcn_bitop3_b16(short a, short b, short c, _Constant unsigned int lut); // bitop3-insts

Performs a bitwise ternary operation using ``lut`` as a truth table. For each
bit position, the corresponding bits of ``a``, ``b``, ``c`` form a 3-bit index
into the 8-bit ``lut``.

GFX12+ Stochastic Rounding Conversion
-------------------------------------

.. code-block:: c

   bf16x2  __builtin_amdgcn_cvt_sr_bf16_f32(bf16x2 old, float val, unsigned int rng, _Constant bool sel);  // f32-to-f16bf16-cvt-sr-insts
   half2   __builtin_amdgcn_cvt_sr_f16_f32(half2 old, float val, unsigned int rng, _Constant bool sel);

- ``old``: Previous packed result.
- ``val``: Input f32 value.
- ``rng``: Random bits for stochastic rounding.
- ``sel``: Selects which half to write.

GFX12+ Scaled Conversion Builtins
---------------------------------

Large family of scaled conversion builtins for FP8/BF8/FP6/BF6/FP4 formats
with scale factors. Representative patterns:

**Unscaled pack to FP6/BF6** (``f16bf16-to-fp6bf6-cvt-scale-insts``):

.. code-block:: c

   uint6 __builtin_amdgcn_cvt_scalef32_pk32_fp6_f16(half32 src, float scale);
   uint6 __builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16(bf16x32 src, float scale);

**Scaled element conversion** (``fp8-cvt-scale-insts`` / ``bf8-cvt-scale-insts``):

.. code-block:: c

   half2  __builtin_amdgcn_cvt_scalef32_f16_fp8(half2 old, int src, float scale, _Constant int byte_sel, _Constant bool word_sel);
   float  __builtin_amdgcn_cvt_scalef32_f32_fp8(int src, float scale, _Constant int byte_sel);
   short2 __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(short2 old, float a, float b, float scale, _Constant bool word_sel);
   float2 __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(unsigned int src, float scale, _Constant bool word_sel);

**Scaled pack FP4** (``fp4-cvt-scale-insts``):

.. code-block:: c

   float2 __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(unsigned int src, float scale, _Constant int nibble_sel);
   unsigned int __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(unsigned int old, float a, float b, float scale, _Constant int nibble_sel);

**Stochastic rounding scaled** variants add a ``unsigned int rng`` argument
for random bits.

**Scaled pack to FP6/BF6 from f32** (``fp6bf6-cvt-scale-insts``):

.. code-block:: c

   float32  __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(uint6 src, float scale);
   half32   __builtin_amdgcn_cvt_scalef32_pk32_f16_fp6(uint6 src, float scale);

GFX1250+ Only Builtins
======================

Require ``gfx1250-insts`` unless otherwise noted.

Cluster Barrier
---------------

.. code-block:: c

   void __builtin_amdgcn_s_cluster_barrier();

Barrier synchronization across all workgroups in the cluster.

Prefetch
--------

.. code-block:: c

   void __builtin_amdgcn_flat_prefetch(const void *ptr, _Constant int length);   // vmem-pref-insts
   void __builtin_amdgcn_global_prefetch(const void __global *ptr, _Constant int length); // vmem-pref-insts

Monitor Loads
-------------

.. code-block:: c

   int   __builtin_amdgcn_global_load_monitor_b32(int __global *ptr, _Constant int ordering, _Constant int scope);
   int2  __builtin_amdgcn_global_load_monitor_b64(int2 __global *ptr, _Constant int ordering, _Constant int scope);
   int4  __builtin_amdgcn_global_load_monitor_b128(int4 __global *ptr, _Constant int ordering, _Constant int scope);
   int   __builtin_amdgcn_flat_load_monitor_b32(int *ptr, _Constant int ordering, _Constant int scope);
   int2  __builtin_amdgcn_flat_load_monitor_b64(int2 *ptr, _Constant int ordering, _Constant int scope);
   int4  __builtin_amdgcn_flat_load_monitor_b128(int4 *ptr, _Constant int ordering, _Constant int scope);

- ``ordering``: Atomic ordering. Must be valid for a load (no ``release``
  or ``acq_rel``).
- ``scope``: Synchronization scope constant.

Cluster Loads
-------------

Require ``mcast-load-insts,wavefrontsize32``.

.. code-block:: c

   int  __builtin_amdgcn_cluster_load_b32(int __global *ptr, _Constant int flags, int mask);
   int2 __builtin_amdgcn_cluster_load_b64(int2 __global *ptr, _Constant int flags, int mask);
   int4 __builtin_amdgcn_cluster_load_b128(int4 __global *ptr, _Constant int flags, int mask);

Async LDS Load/Store (GFX1250)
------------------------------

.. code-block:: c

   void __builtin_amdgcn_cluster_load_async_to_lds_b{8,32,64,128}(T __global *src, T __local *dst, _Constant int flags, _Constant int offset, int mask);
   void __builtin_amdgcn_global_load_async_to_lds_b{8,32,64,128}(T __global *src, T __local *dst, _Constant int flags, _Constant int offset);
   void __builtin_amdgcn_global_store_async_from_lds_b{8,32,64,128}(T __global *dst, T __local *src, _Constant int flags, _Constant int offset);

DS Atomic Barriers
------------------

.. code-block:: c

   void __builtin_amdgcn_ds_atomic_async_barrier_arrive_b64(long __local *ptr);
   long __builtin_amdgcn_ds_atomic_barrier_arrive_rtn_b64(long __local *ptr, long val);

Tensor Load/Store
-----------------

.. code-block:: c

   void __builtin_amdgcn_tensor_load_to_lds(int4 coords, int8 desc, int4 slice0, int4 slice1, _Constant int flags);
   void __builtin_amdgcn_tensor_load_to_lds_d2(int4 coords, int8 desc, _Constant int flags);
   void __builtin_amdgcn_tensor_store_from_lds(int4 coords, int8 desc, int4 slice0, int4 slice1, _Constant int flags);
   void __builtin_amdgcn_tensor_store_from_lds_d2(int4 coords, int8 desc, _Constant int flags);

Transpose Loads (GFX1250)
-------------------------

Various transpose load builtins for different data widths, from both global
and LDS address spaces. These are used to load matrix fragments in transposed
layout for WMMA operations.

.. code-block:: c

   int2   __builtin_amdgcn_global_load_tr4_b64_v2i32(int2 __global *ptr);     // transpose-load-f4f6-insts,wavefrontsize32
   int2   __builtin_amdgcn_global_load_tr8_b64_v2i32(int2 __global *ptr);     // gfx1250-insts,wavefrontsize32
   int3   __builtin_amdgcn_global_load_tr6_b96_v3i32(int3 __global *ptr);     // transpose-load-f4f6-insts,wavefrontsize32
   short8 __builtin_amdgcn_global_load_tr16_b128_v8i16(short8 __global *ptr); // gfx1250-insts,wavefrontsize32
   // ... also half8, bf16x8 variants
   // DS (LDS) variants:
   int2   __builtin_amdgcn_ds_load_tr4_b64_v2i32(int2 __local *ptr);
   // ... etc.

GFX950 also has DS-only transpose reads:

.. code-block:: c

   int2   __builtin_amdgcn_ds_read_tr4_b64_v2i32(int2 __local *ptr);   // gfx950-insts
   int3   __builtin_amdgcn_ds_read_tr6_b96_v3i32(int3 __local *ptr);
   int2   __builtin_amdgcn_ds_read_tr8_b64_v2i32(int2 __local *ptr);
   short4 __builtin_amdgcn_ds_read_tr16_b64_v4i16(short4 __local *ptr);
   // ... also half4, bf16x4 variants

Miscellaneous GFX1250 Builtins
------------------------------

.. code-block:: c

   void __builtin_amdgcn_s_setprio_inc_wg(_Constant short prio);  // setprio-inc-wg-inst
   void __builtin_amdgcn_s_monitor_sleep(_Constant short duration);
   void __builtin_amdgcn_s_wakeup_barrier(void *bar);             // s-wakeup-barrier-inst
   void __builtin_amdgcn_s_wait_asynccnt(_Constant unsigned short count);
   void __builtin_amdgcn_s_wait_tensorcnt(_Constant unsigned short count);

GFX1250 Math Builtins
---------------------

.. code-block:: c

   float  __builtin_amdgcn_tanhf(float x);    // tanh-insts
   __fp16 __builtin_amdgcn_tanhh(__fp16 x);   // tanh-insts
   __bf16 __builtin_amdgcn_tanh_bf16(__bf16 x);  // bf16-trans-insts
   __bf16 __builtin_amdgcn_rcp_bf16(__bf16 x);
   __bf16 __builtin_amdgcn_sqrt_bf16(__bf16 x);
   __bf16 __builtin_amdgcn_rsq_bf16(__bf16 x);
   __bf16 __builtin_amdgcn_log_bf16(__bf16 x);
   __bf16 __builtin_amdgcn_exp2_bf16(__bf16 x);
   __bf16 __builtin_amdgcn_sin_bf16(__bf16 x);
   __bf16 __builtin_amdgcn_cos_bf16(__bf16 x);

Hardware bf16 transcendental functions. All require ``bf16-trans-insts``
(except tanh which requires ``tanh-insts``).

GFX1250 Packed Arithmetic
-------------------------

.. code-block:: c

   unsigned short __builtin_amdgcn_ashr_pk_i8_i32(unsigned int a, unsigned int b, unsigned int c);  // ashr-pk-insts
   unsigned short __builtin_amdgcn_ashr_pk_u8_i32(unsigned int a, unsigned int b, unsigned int c);

   unsigned short __builtin_amdgcn_sat_pk4_i4_i8(unsigned int src);  // gfx1250-insts
   unsigned short __builtin_amdgcn_sat_pk4_u4_u8(unsigned int src);

GFX1250 Add-Min/Max
-------------------

.. code-block:: c

   int          __builtin_amdgcn_add_max_i32(int a, int b, int c, _Constant bool clamp);    // add-min-max-insts
   unsigned int __builtin_amdgcn_add_max_u32(unsigned int a, unsigned int b, unsigned int c, _Constant bool clamp);
   int          __builtin_amdgcn_add_min_i32(int a, int b, int c, _Constant bool clamp);
   unsigned int __builtin_amdgcn_add_min_u32(unsigned int a, unsigned int b, unsigned int c, _Constant bool clamp);

Computes ``max(a + b, c)`` (add_max) or ``min(a + b, c)`` (add_min).
The ``clamp`` flag saturates the result.

Packed 16-bit variants (require ``pk-add-min-max-insts``):

.. code-block:: c

   short2  __builtin_amdgcn_pk_add_max_i16(short2 a, short2 b, short2 c, _Constant bool clamp);
   ushort2 __builtin_amdgcn_pk_add_max_u16(ushort2 a, ushort2 b, ushort2 c, _Constant bool clamp);
   short2  __builtin_amdgcn_pk_add_min_i16(short2 a, short2 b, short2 c, _Constant bool clamp);
   ushort2 __builtin_amdgcn_pk_add_min_u16(ushort2 a, ushort2 b, ushort2 c, _Constant bool clamp);

GFX1250 Permutation
-------------------

Require ``tensor-cvt-lut-insts``.

.. code-block:: c

   uint2 __builtin_amdgcn_perm_pk16_b4_u4(unsigned int src, unsigned int idx, uint2 lut);
   uint3 __builtin_amdgcn_perm_pk16_b6_u4(unsigned int src, unsigned long idx, uint2 lut);
   uint4 __builtin_amdgcn_perm_pk16_b8_u4(unsigned long src, unsigned long idx, uint2 lut);

GFX1250 Scaled Conversions
--------------------------

Additional ``cvt_scale_pk*`` builtins for FP8/BF8/FP6/BF6/FP4 conversions
with scaling, specific to GFX1250:

.. code-block:: c

   half8  __builtin_amdgcn_cvt_scale_pk8_f16_fp8(uint2 src, unsigned int scale, _Constant unsigned int byte_sel);
   bf16x8 __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(uint2 src, unsigned int scale, _Constant unsigned int byte_sel);
   // ... similar for bf8, fp4 source types, and f32 output

**Restriction on ``byte_sel``**: Must be a compile-time constant in the
range [0, 15].

GFX1250 Cooperative Atomics
---------------------------

128-byte cooperative atomic load/store operations. Require
``gfx1250-insts,wavefrontsize32``.

.. code-block:: c

   // 32x4B = 128 bytes, 1 int per lane, 32 lanes
   int  __builtin_amdgcn_cooperative_atomic_load_32x4B(int *ptr, _Constant int ordering, const char *scope);
   void __builtin_amdgcn_cooperative_atomic_store_32x4B(int *ptr, int val, _Constant int ordering, const char *scope);

   // 16x8B = 128 bytes, 2 ints per lane, 16 lanes
   int2 __builtin_amdgcn_cooperative_atomic_load_16x8B(int2 *ptr, _Constant int ordering, const char *scope);
   void __builtin_amdgcn_cooperative_atomic_store_16x8B(int2 *ptr, int2 val, _Constant int ordering, const char *scope);

   // 8x16B = 128 bytes, 4 ints per lane, 8 lanes
   int4 __builtin_amdgcn_cooperative_atomic_load_8x16B(int4 *ptr, _Constant int ordering, const char *scope);
   void __builtin_amdgcn_cooperative_atomic_store_8x16B(int4 *ptr, int4 val, _Constant int ordering, const char *scope);

**Restrictions**:

- ``ptr`` must be in **global** or **generic** address space.
- ``ordering``: Must be a valid C11 atomic ordering. Loads cannot use
  ``release`` or ``acq_rel``. Stores cannot use ``acquire`` or ``acq_rel``.
- ``scope``: Must be a **string literal** specifying a valid synchronization
  scope.

Image Builtins
==============

Image builtins provide access to texture image load, store, and sample
operations. All require ``image-insts`` or ``extended-image-insts``.

The builtin names encode the operation, dimensionality, return type, and
coordinate type:
``__builtin_amdgcn_image_<op>_<dim>_<rettype>_<coordtype>``.

Image Load
----------

.. code-block:: c

   float4 __builtin_amdgcn_image_load_<dim>_v4f32_i32(int dmask, int coord..., __amdgpu_texture_t rsrc, int texfailctrl, int cachepolicy);
   half4  __builtin_amdgcn_image_load_<dim>_v4f16_i32(int dmask, int coord..., __amdgpu_texture_t rsrc, int texfailctrl, int cachepolicy);

Supported dimensions: ``1d``, ``1darray``, ``2d``, ``2darray``, ``3d``,
``cube``. Mip variants: ``mip_1d``, ``mip_1darray``, ``mip_2d``,
``mip_2darray``, ``mip_3d``, ``mip_cube``. Some 2D variants also have
scalar ``float``/``f32`` return.

Arguments:

- ``dmask``: Data mask (constant) selecting which channels to load.
- ``coord...``: Integer coordinates. Count depends on dimensionality
  (1D=1, 2D=2, 3D=3, array=+1, mip=+1).
- ``rsrc``: Texture resource descriptor (``__amdgpu_texture_t``, constant).
- ``texfailctrl``: Texture fail control (constant).
- ``cachepolicy``: Cache policy (constant).

**Restriction**: ``dmask``, ``rsrc``, ``texfailctrl``, and ``cachepolicy``
must all be compile-time constants.

Image Store
-----------

.. code-block:: c

   void __builtin_amdgcn_image_store_<dim>_v4f32_i32(float4 data, int dmask, int coord..., __amdgpu_texture_t rsrc, int texfailctrl, int cachepolicy);
   void __builtin_amdgcn_image_store_<dim>_v4f16_i32(half4 data, int dmask, int coord..., ...);

Same dimensionality and mip variants as loads. The ``data`` argument precedes
the other arguments.

**Restriction**: ``dmask``, ``rsrc``, ``texfailctrl``, and ``cachepolicy``
must all be compile-time constants.

Image Sample
------------

.. code-block:: c

   float4 __builtin_amdgcn_image_sample_<dim>_v4f32_f32(int dmask, float coord..., __amdgpu_texture_t rsrc, int4 sampler, bool unorm, int texfailctrl, int cachepolicy);

Requires ``image-insts``. Supported dimensions: ``1d``, ``1darray``, ``2d``,
``2darray``, ``3d``, ``cube``. Also scalar float return for 2D variants.

Additional arguments compared to image_load:

- ``sampler``: Sampler state (``int4``).
- ``unorm``: If true, coordinates are unnormalized.

Sample Variants
^^^^^^^^^^^^^^^

Require ``extended-image-insts``:

- **``sample_lz``**: Sample at LOD zero. Same args as ``sample``.
- **``sample_l``**: Sample at explicit LOD. Adds one ``float lod`` coordinate.
- **``sample_d``**: Sample with explicit derivatives. Adds derivative
  coordinates (count depends on dimensionality: 2 for 1D, 4 for 2D, 6 for
  3D).
- **``gather4_lz_2d``**: 4-sample gather at LOD zero (2D only).

R600 Builtins
=============

These builtins are for the older R600/Northern Islands GPU targets.

.. code-block:: c

   unsigned char __constant * __builtin_r600_implicitarg_ptr();

   unsigned int __builtin_r600_read_tgid_x();
   unsigned int __builtin_r600_read_tgid_y();
   unsigned int __builtin_r600_read_tgid_z();
   unsigned int __builtin_r600_read_tidig_x();
   unsigned int __builtin_r600_read_tidig_y();
   unsigned int __builtin_r600_read_tidig_z();

   double __builtin_r600_recipsqrt_ieee(double x);
   float  __builtin_r600_recipsqrt_ieeef(float x);

Miscellaneous
=============

.. code-block:: c

   unsigned int __builtin_amdgcn_prng_b32(unsigned int src);  // prng-inst

Pseudo-random number generation.

.. code-block:: c

   void __builtin_amdgcn_s_ttracedata_imm(_Constant short data);  // gfx10-insts

Write immediate data to thread trace.

GFX950 Scaled Conversion Builtins
---------------------------------

.. code-block:: c

   uint6 __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(float16 a, float16 b, float scale);  // gfx950-insts
   uint6 __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(float16 a, float16 b, float scale);

Converts two sets of 16 f32 values to packed FP6/BF6 with scale.
