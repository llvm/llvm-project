// COM: HSV-009 / PLAT-205406 regression: this is the RCCL AllReduce crash
// COM: scenario reduced to comgr lit form. On the 0708 llvmprstack build, RCCL
// COM: device functions (e.g. runTreeUpDown) contain ds_*_2addr sites that sit
// COM: far (> s_branch's +-128 KB reach) from the appended trampoline pool in
// COM: the ~225 MB fatbin. Taking the far path emitted an s_add_pc_i64 long
// COM: branch whose BACKWARD branch-back corrupts wave state on gfx1250 A0,
// COM: producing a GPU memory fault in ncclDevKernel_Generic_4 (0/10 runs).
// COM:
// COM: Required DS2 rewrites use the safe SCC-neutral scratch-SGPR return. The
// COM: two kernels below force the far path for a 2-address load and store and
// COM: verify both are split instead of returning unsafe B0 instructions.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// COM: Case 1 (2-address load, RCCL's pattern).
// DISASM-LABEL: <test_ds2addr_far_load>:
// DISASM-NEXT: s_add_pc_i64

// COM: Case 2 (2-address store).
// DISASM-LABEL: <test_ds2addr_far_store>:
// DISASM-NEXT: s_add_pc_i64

// DISASM-NOT: ds_load_2addr
// DISASM-NOT: ds_store_2addr
// DISASM: ds_load_b64 v[0:1], v4 offset:512
// DISASM-NEXT: ds_load_b64 v[2:3], v4 offset:1024
// DISASM: s_get_pc_i64 s[2:3]
// DISASM-NEXT: s_add_nc_u64 s[2:3]
// DISASM-NEXT: s_set_pc_i64 s[2:3]
// DISASM: ds_store_b32 v2, v0 offset:256
// DISASM-NEXT: ds_store_b32 v2, v1 offset:768
// DISASM: s_get_pc_i64 s[2:3]
// DISASM-NEXT: s_add_nc_u64 s[2:3]
// DISASM-NEXT: s_set_pc_i64 s[2:3]

// METADATA: .name:           test_ds2addr_far_load
// METADATA: .sgpr_count:     6
// METADATA: .name:           test_ds2addr_far_store
// METADATA: .sgpr_count:     6

// COM: Idempotency: rewriting the split output again is a no-op.
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

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_ds2addr_far_load
      .symbol: test_ds2addr_far_load.kd
      .sgpr_count: 1
      .vgpr_count: 5
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_ds2addr_far_store
      .symbol: test_ds2addr_far_store.kd
      .sgpr_count: 1
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
