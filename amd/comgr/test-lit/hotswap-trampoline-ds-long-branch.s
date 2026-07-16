// COM: HSV-009 / PLAT-205406 regression: this is the RCCL AllReduce crash
// COM: scenario reduced to comgr lit form. On the 0708 llvmprstack build, RCCL
// COM: device functions (e.g. runTreeUpDown) contain ds_*_2addr sites that sit
// COM: far (> s_branch's +-128 KB reach) from the appended trampoline pool in
// COM: the ~225 MB fatbin. Taking the far path emitted an s_add_pc_i64 long
// COM: branch. On affected gfx1250 parts, even one execution can corrupt SGPR
// COM: forwarding in another wave on the same SIMD, causing wrong results,
// COM: hangs, or a GPU page fault.
// COM:
// COM: Fix: use the gfx12 SCC-neutral SGPR-backed set-PC sequence on both
// COM: edges. Merge adjacent sites and bounded straight-line neighbors when
// COM: they provide enough source bytes; otherwise short-branch through safe
// COM: NOP padding.
// COM: No path emits the broken instruction.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// COM: Case 1 (2-address load, RCCL's pattern).
// DISASM-NOT: s_add_pc_i64
// DISASM-LABEL: <test_ds2addr_far_load>:
// DISASM-NEXT: s_branch

// COM: Case 2 (adjacent sites coalesced into a set-PC forward gateway).
// DISASM-LABEL: <test_ds2addr_far_adjacent>:
// DISASM-NEXT: s_get_pc_i64 s[2:3]
// DISASM-NEXT: s_add_nc_u64 s[2:3], s[2:3],
// DISASM-NEXT: s_set_pc_i64 s[2:3]

// COM: Case 3 (an interior direct-branch target must prevent coalescing).
// DISASM-LABEL: <test_ds2addr_far_branch_target>:
// DISASM-NEXT: s_cbranch_scc1
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_branch

// COM: Case 4 (a direct-call target is also an interior entry point).
// DISASM-LABEL: <test_ds2addr_far_call_target>:
// DISASM-NEXT: s_call_i64
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_branch

// COM: Case 5 (2-address store).
// DISASM-LABEL: <test_ds2addr_far_store>:
// DISASM-NEXT: s_branch

// DISASM-NOT: ds_load_2addr
// DISASM-NOT: ds_store_2addr
// DISASM: ds_load_b64 v[8:9], v12 offset:2560
// DISASM-NEXT: ds_load_b64 v[10:11], v12 offset:3072
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: ds_load_b64 v[12:13], v16 offset:3584
// DISASM-NEXT: ds_load_b64 v[14:15], v16 offset:4096
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_get_pc_i64 s[2:3]
// DISASM-NEXT: s_add_nc_u64 s[2:3], s[2:3],
// DISASM-NEXT: s_set_pc_i64 s[2:3]
// DISASM: ds_store_b32 v2, v0 offset:256
// DISASM-NEXT: ds_store_b32 v2, v1 offset:768
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_get_pc_i64 s[2:3]
// DISASM-NEXT: s_add_nc_u64 s[2:3], s[2:3],
// DISASM-NEXT: s_set_pc_i64 s[2:3]

// METADATA: .name:           test_ds2addr_far_load
// METADATA: .sgpr_count:     6
// METADATA: .name:           test_ds2addr_far_store
// METADATA: .sgpr_count:     6
// METADATA: .name:           test_ds2addr_far_adjacent
// METADATA: .sgpr_count:     6
// METADATA: .name:           test_ds2addr_far_branch_target
// METADATA: .sgpr_count:     6
// METADATA: .name:           test_ds2addr_far_call_target
// METADATA: .sgpr_count:     6

// COM: Idempotency: rewriting the patched output again is a no-op.
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

.globl test_ds2addr_far_adjacent
.p2align 8
.type test_ds2addr_far_adjacent,@function
test_ds2addr_far_adjacent:
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
  ds_load_2addr_stride64_b64 v[4:7], v8 offset0:3 offset1:4
  ds_load_2addr_stride64_b64 v[8:11], v12 offset0:5 offset1:6
  ds_load_2addr_stride64_b64 v[12:15], v16 offset0:7 offset1:8
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_ds2addr_far_adjacent_end:
.size test_ds2addr_far_adjacent, .Ltest_ds2addr_far_adjacent_end-test_ds2addr_far_adjacent

.globl test_ds2addr_far_branch_target
.p2align 8
.type test_ds2addr_far_branch_target,@function
test_ds2addr_far_branch_target:
  s_cbranch_scc1 .Lsecond_ds2addr
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
.Lsecond_ds2addr:
  ds_load_2addr_stride64_b64 v[4:7], v8 offset0:3 offset1:4
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_ds2addr_far_branch_target_end:
.size test_ds2addr_far_branch_target, .Ltest_ds2addr_far_branch_target_end-test_ds2addr_far_branch_target

.globl test_ds2addr_far_call_target
.p2align 8
.type test_ds2addr_far_call_target,@function
test_ds2addr_far_call_target:
  s_call_i64 s[0:1], .Lcalled_ds2addr
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
.Lcalled_ds2addr:
  ds_load_2addr_stride64_b64 v[4:7], v8 offset0:3 offset1:4
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_ds2addr_far_call_target_end:
.size test_ds2addr_far_call_target, .Ltest_ds2addr_far_call_target_end-test_ds2addr_far_call_target

.globl test_ds2addr_far_store
.p2align 8
.type test_ds2addr_far_store,@function
test_ds2addr_far_store:
  ds_store_2addr_stride64_b32 v2, v0, v1 offset0:1 offset1:3
  s_wait_dscnt 0x0
  s_endpgm
  // ~160 KB of non-NOP filler (forms no usable sled) so the appended trampoline
  // pool is beyond s_branch's +-128 KB reach from both kernels above, forcing
  // the far (long-branch) path.
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

.amdhsa_kernel test_ds2addr_far_adjacent
  .amdhsa_next_free_vgpr 17
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds2addr_far_branch_target
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel test_ds2addr_far_call_target
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 2
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
    - .name: test_ds2addr_far_adjacent
      .symbol: test_ds2addr_far_adjacent.kd
      .sgpr_count: 1
      .vgpr_count: 17
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_ds2addr_far_branch_target
      .symbol: test_ds2addr_far_branch_target.kd
      .sgpr_count: 1
      .vgpr_count: 9
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: test_ds2addr_far_call_target
      .symbol: test_ds2addr_far_call_target.kd
      .sgpr_count: 2
      .vgpr_count: 9
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
