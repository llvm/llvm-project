// COM: Test that zero bytes inside a function symbol are not treated as a
// COM: zero-fill NOP sled for v_cvt_pk_fp8_f32 literal-source expansion.
// COM:
// COM: The zero block below is executable .text covered by the function's
// COM: STT_FUNC range, so the rewriter must leave it alone and append a
// COM: trampoline instead of branching into the zero bytes.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump %t.out.elf --check-idempotent 2>&1 \
// RUN:   | %FileCheck --check-prefix=API %s
// API: hotswap: growWithTrampolines: appended 1 trampoline
// API-SAME: grew ELF
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_cvt_pk_fp8_zero_in_function
.type test_cvt_pk_fp8_zero_in_function,@function
test_cvt_pk_fp8_zero_in_function:
  v_cvt_pk_fp8_f32 v4, 0x477f0000, 0x477f0000 clamp
  s_endpgm
  .zero 768
.Ltest_cvt_pk_fp8_zero_in_function_end:
.size test_cvt_pk_fp8_zero_in_function, .Ltest_cvt_pk_fp8_zero_in_function_end-test_cvt_pk_fp8_zero_in_function

// DISASM-LABEL: <test_cvt_pk_fp8_zero_in_function>:
// DISASM-NEXT:  s_branch
// DISASM-NEXT:  s_nop
// DISASM-NEXT:  s_nop
// DISASM-NEXT:  s_endpgm
// DISASM:       v_mov_b32{{.*}}0x477f0000
// DISASM:       v_lshl_or_b32
// DISASM:       s_branch{{.*}}<test_cvt_pk_fp8_zero_in_function+0xc>

.rodata
.p2align 8
.amdhsa_kernel test_cvt_pk_fp8_zero_in_function
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_cvt_pk_fp8_zero_in_function
      .symbol: test_cvt_pk_fp8_zero_in_function.kd
      .sgpr_count: 2
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
