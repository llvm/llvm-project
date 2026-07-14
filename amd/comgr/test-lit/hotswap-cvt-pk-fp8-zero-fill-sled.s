// COM: Test v_cvt_pk_fp8_f32 CLAMP=1 literal sources with zero-fill padding.
// COM:
// COM: Zero-filled alignment gaps between function-symbol ranges are writable
// COM: .text bytes, but they are not decoded s_nop instructions inside the
// COM: source function. The rewriter should leave them alone and append a
// COM: trampoline instead of branching into the zero bytes.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump %t.out.elf --check-idempotent \
// RUN:   | %FileCheck --check-prefix=API %s
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_cvt_pk_fp8_zero_fill_source
.p2align 8
.type test_cvt_pk_fp8_zero_fill_source,@function
test_cvt_pk_fp8_zero_fill_source:
  v_cvt_pk_fp8_f32 v4, 0x477f0000, 0x477f0000 clamp
  s_endpgm
.Ltest_cvt_pk_fp8_zero_fill_source_end:
.size test_cvt_pk_fp8_zero_fill_source, .Ltest_cvt_pk_fp8_zero_fill_source_end-test_cvt_pk_fp8_zero_fill_source

.globl test_cvt_pk_fp8_zero_fill_sled
test_cvt_pk_fp8_zero_fill_sled:
  .zero 768

.globl test_cvt_pk_fp8_zero_fill_after
.p2align 8
.type test_cvt_pk_fp8_zero_fill_after,@function
test_cvt_pk_fp8_zero_fill_after:
  s_endpgm
.Ltest_cvt_pk_fp8_zero_fill_after_end:
.size test_cvt_pk_fp8_zero_fill_after, .Ltest_cvt_pk_fp8_zero_fill_after_end-test_cvt_pk_fp8_zero_fill_after

// DISASM-LABEL: <test_cvt_pk_fp8_zero_fill_source>:
// COM: The cvt is replaced by a forward branch into the appended trampoline
// COM: pool (a fresh vaddr above .text); the pool section is unnamed, so
// COM: objdump prints no symbol annotation for the target.
// DISASM-NEXT:  s_branch
// DISASM-NEXT:  s_nop
// DISASM-NEXT:  s_nop
// DISASM-NEXT:  s_endpgm
// DISASM-LABEL: <test_cvt_pk_fp8_zero_fill_sled>:
// DISASM-NOT:   s_mov_b32
// DISASM-NOT:   v_mov_b32{{.*}}0x477f0000
// DISASM-NOT:   v_lshl_or_b32
// DISASM-LABEL: <test_cvt_pk_fp8_zero_fill_after>:
// DISASM-NEXT:  s_endpgm
// COM: Trampoline body lives in the appended pool section (fresh vaddr above
// COM: .text), so objdump emits a section header here -- DISASM, not -NEXT.
// DISASM:       s_mov_b32
// DISASM-NEXT:  v_mov_b32{{.*}}0x477f0000
// DISASM-NEXT:  v_mov_b32{{.*}}0x477f0000
// DISASM-NEXT:  v_and_b32{{.*}}0x7fffffff
// DISASM:       v_lshl_or_b32
// DISASM-NEXT:  v_bfi_b32 v4,
// DISASM-NEXT:  s_mov_b32
// DISASM-NEXT:  s_branch{{.*}}<test_cvt_pk_fp8_zero_fill_source+0xc>

.rodata
.p2align 8
.amdhsa_kernel test_cvt_pk_fp8_zero_fill_source
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
.amdhsa_kernel test_cvt_pk_fp8_zero_fill_after
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_cvt_pk_fp8_zero_fill_source
      .symbol: test_cvt_pk_fp8_zero_fill_source.kd
      .sgpr_count: 2
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_cvt_pk_fp8_zero_fill_after
      .symbol: test_cvt_pk_fp8_zero_fill_after.kd
      .sgpr_count: 2
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
