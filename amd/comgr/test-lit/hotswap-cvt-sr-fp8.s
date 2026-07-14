// COM: Test v_cvt_sr_fp8_f32 CLAMP=1 (E5M3) stochastic-round conversion patch.
// COM:
// COM: Creates a minimal gfx1250 code object containing v_cvt_sr_fp8_f32
// COM: with clamp (E5M3 mode), runs the hotswap rewrite, and verifies the
// COM: replacement sequence covers: NaN detection, stochastic noise
// COM: addition, F32->F16->UE5M3 conversion, overflow clamping, NaN
// COM: override, and byte merge.
// COM:
// COM: Companion tests:
// COM:   hotswap-cvt-fp8-modifiers.s - source modifier variants
// COM:   hotswap-cvt-fp8-nosled.s    - trampoline fallback path
// COM:   hotswap-cvt-fp8-multi.s     - multi-site stacking

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump %t.out.elf --check-idempotent \
// RUN:   | %FileCheck --check-prefix=API %s
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=BYTE0 %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=BYTE2 %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=NOCLAMP %s

// ---- Kernel 1: CLAMP=1, byte_sel=0 (should be patched) -----------------------
//
// COM: Original site is replaced with s_branch forward to trampoline.
// COM: Trampoline body: VCC save, NaN detection, clamp negative, stochastic
// COM: noise injection, F32->F16->UE5M3 conversion, overflow clamping,
// COM: NaN override, byte merge (single bfi for byte_sel=0), VCC restore.

// BYTE0-LABEL: <test_cvt_sr_fp8_byte0>:
// BYTE0:       s_branch
// COM: --- VCC save ---
// BYTE0:       s_mov_b32
// COM: --- NaN detection ---
// BYTE0-NEXT:  v_and_b32{{.*}}0x7fffffff, v1
// BYTE0-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// BYTE0-NEXT:  s_mov_b32
// COM: --- Clamp negative ---
// BYTE0-NEXT:  v_max_num_f32{{.*}}, 0, v1
// COM: --- Stochastic noise injection ---
// BYTE0-NEXT:  v_lshrrev_b32{{.*}}, 12, v2
// BYTE0-NEXT:  v_add
// COM: --- High-range direct path ---
// BYTE0-NEXT:  v_cmp_le_f32{{.*}}0x47800000
// BYTE0-NEXT:  s_mov_b32
// BYTE0-NEXT:  v_mul_f32{{.*}}0x39000000
// BYTE0-NEXT:  v_cvt_floor_i32_f32
// BYTE0-NEXT:  v_add{{.*}}0xf0
// BYTE0-NEXT:  v_min_u32{{.*}}0xfe
// COM: --- F32 -> F16 -> UE5M3 ---
// BYTE0-NEXT:  v_cvt_f16_f32
// BYTE0-NEXT:  v_bfe_u32
// COM: --- Overflow clamp and high-range select ---
// BYTE0-NEXT:  v_min_u32{{.*}}0xfe
// BYTE0-NEXT:  s_mov_b32
// BYTE0-NEXT:  v_cndmask_b32
// COM: --- NaN override ---
// BYTE0-NEXT:  s_mov_b32
// BYTE0-NEXT:  v_mov_b32
// BYTE0-NEXT:  v_cndmask_b32
// COM: --- Byte merge (byte_sel=0: single bfi) ---
// BYTE0-NEXT:  v_bfi_b32 v0,
// COM: --- VCC restore ---
// BYTE0-NEXT:  s_mov_b32

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_cvt_sr_fp8_byte0
.p2align 8
.type test_cvt_sr_fp8_byte0,@function
test_cvt_sr_fp8_byte0:
  v_cvt_sr_fp8_f32 v0, v1, v2 clamp
  s_endpgm
.Ltest_cvt_sr_fp8_byte0_end:
.size test_cvt_sr_fp8_byte0, .Ltest_cvt_sr_fp8_byte0_end-test_cvt_sr_fp8_byte0

// ---- Kernel 2: CLAMP=1, byte_sel=2 (raw encoding for OPSEL[3:2]) -------------
//
// COM: Same conversion sequence as byte_sel=0, but final merge uses shift + bfi
// COM: to write the result into byte 2 of vdst.

// BYTE2-LABEL: <test_cvt_sr_fp8_byte2>:
// BYTE2:       s_branch
// COM: --- VCC save + NaN detection (anchor on unique src v6) ---
// BYTE2:       v_and_b32{{.*}}0x7fffffff, v6
// BYTE2-NEXT:  v_cmp_lt_u32{{.*}}0x7f800000
// BYTE2-NEXT:  s_mov_b32
// COM: --- Clamp negative ---
// BYTE2-NEXT:  v_max_num_f32{{.*}}, 0, v6
// COM: --- Stochastic noise injection ---
// BYTE2-NEXT:  v_lshrrev_b32{{.*}}, 12, v7
// BYTE2-NEXT:  v_add
// COM: --- High-range direct path ---
// BYTE2-NEXT:  v_cmp_le_f32{{.*}}0x47800000
// BYTE2-NEXT:  s_mov_b32
// BYTE2-NEXT:  v_mul_f32{{.*}}0x39000000
// BYTE2-NEXT:  v_cvt_floor_i32_f32
// BYTE2-NEXT:  v_add{{.*}}0xf0
// BYTE2-NEXT:  v_min_u32{{.*}}0xfe
// COM: --- F32 -> F16 -> UE5M3 ---
// BYTE2-NEXT:  v_cvt_f16_f32
// BYTE2-NEXT:  v_bfe_u32
// COM: --- Overflow clamp and high-range select ---
// BYTE2-NEXT:  v_min_u32{{.*}}0xfe
// BYTE2-NEXT:  s_mov_b32
// BYTE2-NEXT:  v_cndmask_b32
// COM: --- NaN override ---
// BYTE2-NEXT:  s_mov_b32
// BYTE2-NEXT:  v_mov_b32
// BYTE2-NEXT:  v_cndmask_b32
// COM: --- Byte merge (byte_sel=2: shift + bfi) ---
// BYTE2-NEXT:  v_lshlrev_b32
// BYTE2-NEXT:  v_bfi_b32 v5,
// COM: --- VCC restore ---
// BYTE2-NEXT:  s_mov_b32

.globl test_cvt_sr_fp8_byte2
.p2align 8
.type test_cvt_sr_fp8_byte2,@function
test_cvt_sr_fp8_byte2:
  // v_cvt_sr_fp8_f32 v5, v6, v7 clamp byte_sel=2
  // dword0 = 0xD76BC005 (CLAMP=1, OPSEL[3]=1, OPSEL[2]=0 -> byte_sel=2, vdst=v5)
  // dword1 = 0x02020F06 (src0=v6, src1=v7, no modifiers)
  .long 0xD76BC005
  .long 0x02020F06
  s_endpgm
.Ltest_cvt_sr_fp8_byte2_end:
.size test_cvt_sr_fp8_byte2, .Ltest_cvt_sr_fp8_byte2_end-test_cvt_sr_fp8_byte2

// ---- Kernel 3: no clamp (should NOT be patched) -------------------------------

// NOCLAMP-LABEL: <test_cvt_sr_fp8_noclamp>:
// NOCLAMP-NEXT:  v_cvt_sr_fp8_f32

.globl test_cvt_sr_fp8_noclamp
.p2align 8
.type test_cvt_sr_fp8_noclamp,@function
test_cvt_sr_fp8_noclamp:
  v_cvt_sr_fp8_f32 v10, v11, v12
  s_endpgm
.Ltest_cvt_sr_fp8_noclamp_end:
.size test_cvt_sr_fp8_noclamp, .Ltest_cvt_sr_fp8_noclamp_end-test_cvt_sr_fp8_noclamp

.rodata
.p2align 8
.amdhsa_kernel test_cvt_sr_fp8_byte0
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_sr_fp8_byte2
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_sr_fp8_noclamp
  .amdhsa_next_free_vgpr 13
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_cvt_sr_fp8_byte0
      .symbol: test_cvt_sr_fp8_byte0.kd
      .sgpr_count: 2
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_sr_fp8_byte2
      .symbol: test_cvt_sr_fp8_byte2.kd
      .sgpr_count: 2
      .vgpr_count: 8
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_sr_fp8_noclamp
      .symbol: test_cvt_sr_fp8_noclamp.kd
      .sgpr_count: 2
      .vgpr_count: 13
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
