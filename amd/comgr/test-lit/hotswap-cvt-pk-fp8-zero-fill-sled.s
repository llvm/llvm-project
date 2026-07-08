// COM: Test v_cvt_pk_fp8_f32 CLAMP=1 literal sources using zero-fill padding.
// COM:
// COM: Generated hipblasLt initializer kernels can leave zero-filled alignment
// COM: gaps between function-symbol ranges. Those gaps are writable .text
// COM: space even though they are not decoded as s_nop instructions, and they
// COM: are often the only local landing zones large enough for the expanded
// COM: literal-source FP8 pack replacement.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --dump %t.out.elf --check-idempotent \
// RUN:   | %FileCheck --check-prefix=API %s
// API: REWRITE: SUCCESS
// API: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=ZERO %s

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

// ZERO-LABEL: <test_cvt_pk_fp8_zero_fill_source>:
// ZERO-NEXT:  s_branch{{.*}}test_cvt_pk_fp8_zero_fill_sled
// ZERO-NEXT:  s_nop
// ZERO-NEXT:  s_nop
// ZERO-NEXT:  s_endpgm
// ZERO-LABEL: <test_cvt_pk_fp8_zero_fill_sled>:
// ZERO:       s_mov_b32
// ZERO-NEXT:  v_mov_b32{{.*}}0x477f0000
// ZERO-NEXT:  v_mov_b32{{.*}}0x477f0000
// ZERO-NEXT:  v_and_b32{{.*}}0x7fffffff
// ZERO:       v_lshl_or_b32
// ZERO-NEXT:  v_bfi_b32 v4,
// ZERO-NEXT:  s_mov_b32
// ZERO-NEXT:  s_branch{{.*}}<test_cvt_pk_fp8_zero_fill_source+0xc>
// ZERO-LABEL: <test_cvt_pk_fp8_zero_fill_after>:
// ZERO-NEXT:  s_endpgm

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
