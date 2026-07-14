// COM: A cyclic destination/source dependency cannot be split into two
// COM: sequential exchanges without scratch VGPRs. Returning the original B0
// COM: DS2 instruction in an A0 object would be unsafe, so rewriting must fail.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck %s

// CHECK: hotswap: error: ds_storexchg_2addr has cyclic destination/source overlap
// CHECK-NOT: hotswap: error: ds_2addr expansion failed
// CHECK: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds_overlap_cyclic
.p2align 8
.type test_ds_overlap_cyclic,@function
test_ds_overlap_cyclic:
  ds_storexchg_2addr_rtn_b64 v[20:23], v24, v[22:23], v[20:21] offset0:0 offset1:1
  s_wait_dscnt 0
  s_endpgm
.Ltest_ds_overlap_cyclic_end:
.size test_ds_overlap_cyclic, .Ltest_ds_overlap_cyclic_end-test_ds_overlap_cyclic

.rodata
.p2align 8
.amdhsa_kernel test_ds_overlap_cyclic
  .amdhsa_next_free_vgpr 25
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
