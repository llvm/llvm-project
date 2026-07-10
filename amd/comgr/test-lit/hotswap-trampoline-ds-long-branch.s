// COM: HSV-009 / PLAT-205406 regression: this is the RCCL AllReduce crash
// COM: scenario reduced to comgr lit form. On the 0708 llvmprstack build, RCCL
// COM: device functions (e.g. runTreeUpDown) contain ds_*_2addr sites that sit
// COM: far (> s_branch's +-128 KB reach) from the appended trampoline pool in
// COM: the ~225 MB fatbin. Taking the far path emitted an s_add_pc_i64 long
// COM: branch whose BACKWARD branch-back corrupts wave state on gfx1250 A0,
// COM: producing a GPU memory fault in ncclDevKernel_Generic_4 (0/10 runs).
// COM:
// COM: Fix: decline far ds_*_2addr sites (leave the original instruction) until
// COM: a scratch-register long branch-back lands. The two kernels below force
// COM: the far path for a 2-addr LOAD and STORE and assert the decline: the
// COM: original ds_*_2addr stays, and neither an s_add_pc_i64 redirect nor the
// COM: single-address split (ds_load_b64 / ds_store_b32) is emitted. The near
// COM: (sled / short-s_branch) ds_*_2addr path is still exercised and split by
// COM: hotswap-trampoline-ds.s; only the far path changes here.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: Case 1 (2-addr LOAD, RCCL's pattern): declined -- original op stays, no
// COM: forward long branch, no single-address split.
// DISASM-LABEL: <test_ds2addr_far_load>:
// DISASM-NEXT: ds_load_2addr_stride64_b64
// DISASM-NOT: s_add_pc_i64
// DISASM-NOT: ds_load_b64

// COM: Case 2 (2-addr STORE): declined the same way.
// DISASM-LABEL: <test_ds2addr_far_store>:
// DISASM-NEXT: ds_store_2addr_stride64_b32
// DISASM-NOT: s_add_pc_i64
// DISASM-NOT: ds_store_b32

// COM: Idempotency: rewriting the declined output again is a no-op.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_ds2addr_far_load
.p2align 8
.type test_ds2addr_far_load,@function
test_ds2addr_far_load:
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_ds2addr_far_load_end:
.size test_ds2addr_far_load, .Ltest_ds2addr_far_load_end-test_ds2addr_far_load

.globl test_ds2addr_far_store
.p2align 8
.type test_ds2addr_far_store,@function
test_ds2addr_far_store:
  ds_store_2addr_stride64_b32 v2, v0, v1 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
  // ~160 KB of non-NOP filler (forms no usable sled) so the appended trampoline
  // pool is beyond s_branch's +-128 KB reach from both kernels above, forcing
  // the far (long-branch) path -- which is declined on gfx1250 A0.
  .rept 40000
    s_mov_b32 s0, s1
  .endr
.Ltest_ds2addr_far_store_end:
.size test_ds2addr_far_store, .Ltest_ds2addr_far_store_end-test_ds2addr_far_store

.rodata
.p2align 8
.amdhsa_kernel test_ds2addr_far_load
  .amdhsa_next_free_vgpr 5
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds2addr_far_store
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel
