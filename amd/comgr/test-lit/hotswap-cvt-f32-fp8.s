// COM: Test v_cvt_f32_fp8 CLAMP=1 (E5M3) UE5M3->F32 unpack conversion patch.
// COM:
// COM: Creates a minimal gfx1250 code object containing v_cvt_f32_fp8
// COM: with clamp (E5M3 mode), runs the hotswap rewrite, and verifies the
// COM: replacement sequence covers: byte extraction, NaN detection, exp-31
// COM: detection, exp-31 direct F32 construction, F16 base path, exp-31
// COM: select, and NaN override.
// COM:
// COM: Companion tests:
// COM:   hotswap-cvt-fp8-bytesel.s  — all 4 byte_sel positions
// COM:   hotswap-cvt-fp8-nosled.s   — trampoline fallback path
// COM:   hotswap-cvt-fp8-multi.s    — multi-site stacking

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
// COM: Trampoline body: VCC save, byte extraction (v_and_b32 0xff),
// COM: NaN detection (v_cmp_eq_u32 0xff), exp-31 detection (v_cmp_lt_u32),
// COM: exp-31 direct F32 construction, F16 base path (v_lshlrev + v_cvt_f32_f16),
// COM: exp-31 select (v_cndmask), NaN override (v_cndmask), VCC restore.

// BYTE0-LABEL: <test_cvt_f32_fp8_byte0>:
// BYTE0:       s_branch
// COM: --- VCC save ---
// BYTE0:       s_mov_b32
// COM: --- Byte extraction (byte_sel=0: v_and_b32) ---
// BYTE0-NEXT:  v_and_b32{{.*}}0xff, v1
// COM: --- NaN detection ---
// BYTE0-NEXT:  v_cmp_eq_u32{{.*}}0xff
// BYTE0-NEXT:  s_mov_b32
// COM: --- Exp-31 detection ---
// BYTE0-NEXT:  v_cmp_lt_u32{{.*}}0xf7
// BYTE0-NEXT:  s_mov_b32
// COM: --- Exp-31 direct F32 construction ---
// BYTE0-NEXT:  v_and_b32{{.*}} 7,
// BYTE0-NEXT:  v_lshlrev_b32{{.*}}, 20
// BYTE0-NEXT:  v_or_b32{{.*}}0x47800000
// COM: --- F16 base path ---
// BYTE0-NEXT:  v_lshlrev_b32{{.*}}, 7
// BYTE0-NEXT:  v_cvt_f32_f16
// COM: --- Exp-31 select ---
// BYTE0-NEXT:  s_mov_b32
// BYTE0-NEXT:  v_cndmask_b32
// COM: --- NaN override ---
// BYTE0-NEXT:  s_mov_b32
// BYTE0-NEXT:  v_mov_b32{{.*}}0x7fa3d000
// BYTE0-NEXT:  v_cndmask_b32{{.*}}v0,
// COM: --- VCC restore ---
// BYTE0-NEXT:  s_mov_b32

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_cvt_f32_fp8_byte0
.p2align 8
.type test_cvt_f32_fp8_byte0,@function
test_cvt_f32_fp8_byte0:
  v_cvt_f32_fp8 v0, v1 clamp
  s_endpgm
.Ltest_cvt_f32_fp8_byte0_end:
.size test_cvt_f32_fp8_byte0, .Ltest_cvt_f32_fp8_byte0_end-test_cvt_f32_fp8_byte0

// ---- Kernel 2: CLAMP=1, byte_sel=2 (v_bfe_u32 extraction) --------------------
//
// COM: Byte extraction for byte_sel=2 uses v_bfe_u32 (offset=16, width=8)
// COM: instead of v_and_b32.

// BYTE2-LABEL: <test_cvt_f32_fp8_byte2>:
// BYTE2:       s_branch
// COM: --- VCC save + Byte extraction (anchor on unique src v6) ---
// BYTE2:       v_bfe_u32{{.*}}v6, 16, 8
// COM: --- NaN detection ---
// BYTE2-NEXT:  v_cmp_eq_u32{{.*}}0xff
// BYTE2-NEXT:  s_mov_b32
// COM: --- Exp-31 detection ---
// BYTE2-NEXT:  v_cmp_lt_u32{{.*}}0xf7
// BYTE2-NEXT:  s_mov_b32
// COM: --- Exp-31 direct F32 construction ---
// BYTE2-NEXT:  v_and_b32{{.*}} 7,
// BYTE2-NEXT:  v_lshlrev_b32{{.*}}, 20
// BYTE2-NEXT:  v_or_b32{{.*}}0x47800000
// COM: --- F16 base path ---
// BYTE2-NEXT:  v_lshlrev_b32{{.*}}, 7
// BYTE2-NEXT:  v_cvt_f32_f16
// COM: --- Exp-31 select ---
// BYTE2-NEXT:  s_mov_b32
// BYTE2-NEXT:  v_cndmask_b32
// COM: --- NaN override ---
// BYTE2-NEXT:  s_mov_b32
// BYTE2-NEXT:  v_mov_b32{{.*}}0x7fa3d000
// BYTE2-NEXT:  v_cndmask_b32{{.*}}v5,
// COM: --- VCC restore ---
// BYTE2-NEXT:  s_mov_b32

.globl test_cvt_f32_fp8_byte2
.p2align 8
.type test_cvt_f32_fp8_byte2,@function
test_cvt_f32_fp8_byte2:
  // v_cvt_f32_fp8 v5, v6 clamp byte_sel=2
  // dword0 = 0xD5EC8805 (CLAMP=1, OPSEL[0]=1 -> byte_sel=2, vdst=v5)
  // dword1 = 0x02010106 (src0=v6, no modifiers)
  .long 0xD5EC8805
  .long 0x02010106
  s_endpgm
.Ltest_cvt_f32_fp8_byte2_end:
.size test_cvt_f32_fp8_byte2, .Ltest_cvt_f32_fp8_byte2_end-test_cvt_f32_fp8_byte2

// ---- Kernel 3: no clamp (should NOT be patched) -------------------------------

// NOCLAMP-LABEL: <test_cvt_f32_fp8_noclamp>:
// NOCLAMP-NEXT:  v_cvt_f32_fp8

.globl test_cvt_f32_fp8_noclamp
.p2align 8
.type test_cvt_f32_fp8_noclamp,@function
test_cvt_f32_fp8_noclamp:
  v_cvt_f32_fp8 v10, v11
  s_endpgm
.Ltest_cvt_f32_fp8_noclamp_end:
.size test_cvt_f32_fp8_noclamp, .Ltest_cvt_f32_fp8_noclamp_end-test_cvt_f32_fp8_noclamp

.rodata
.p2align 8
.amdhsa_kernel test_cvt_f32_fp8_byte0
  .amdhsa_next_free_vgpr 2
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_f32_fp8_byte2
  .amdhsa_next_free_vgpr 7
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_f32_fp8_noclamp
  .amdhsa_next_free_vgpr 12
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_cvt_f32_fp8_byte0
      .symbol: test_cvt_f32_fp8_byte0.kd
      .sgpr_count: 2
      .vgpr_count: 2
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_f32_fp8_byte2
      .symbol: test_cvt_f32_fp8_byte2.kd
      .sgpr_count: 2
      .vgpr_count: 7
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_f32_fp8_noclamp
      .symbol: test_cvt_f32_fp8_noclamp.kd
      .sgpr_count: 2
      .vgpr_count: 12
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
