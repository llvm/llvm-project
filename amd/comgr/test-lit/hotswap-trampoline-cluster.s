// COM: Test HotSwap trampoline patching for cluster_load forms that are not
// COM: demoted in place. Remaining cluster loads must run with M0.wg_mask
// COM: bits [15:0] cleared on A0, while preserving the incoming M0 value for
// COM: surrounding code.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// DISASM-LABEL: <test_cluster_mask>:
// DISASM: s_branch
// DISASM: s_wait_loadcnt
// DISASM: s_branch
// DISASM: s_wait_loadcnt
// DISASM: s_branch
// DISASM: s_wait_loadcnt
// DISASM: s_branch
// DISASM: s_wait_loadcnt
// DISASM: s_branch
// DISASM: s_wait_loadcnt
// DISASM: s_branch
// DISASM: s_wait_loadcnt
// DISASM: s_endpgm
// DISASM: s_mov_b32 [[SCR0:s[0-9]+]], m0
// DISASM-NEXT: s_pack_hh_b32_b16 m0, 0, m0
// DISASM-NEXT: cluster_load_b64 v[{{[0-9:]+}}], v{{[0-9]+}}, s[{{[0-9:]+}}]
// DISASM-NEXT: s_mov_b32 m0, [[SCR0]]
// DISASM-NEXT: s_branch
// DISASM: s_mov_b32 [[SCR1:s[0-9]+]], m0
// DISASM-NEXT: s_pack_hh_b32_b16 m0, 0, m0
// DISASM-NEXT: cluster_load_async_to_lds_b32 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}]
// DISASM-NEXT: s_mov_b32 m0, [[SCR1]]
// DISASM-NEXT: s_branch
// DISASM: s_mov_b32 [[SCR2:s[0-9]+]], m0
// DISASM-NEXT: s_pack_hh_b32_b16 m0, 0, m0
// DISASM-NEXT: cluster_load_b128 v[{{[0-9:]+}}], v{{[0-9]+}}, s[{{[0-9:]+}}] offset:64 scale_offset th:TH_LOAD_NT_HT scope:SCOPE_DEV
// DISASM-NEXT: s_mov_b32 m0, [[SCR2]]
// DISASM-NEXT: s_branch
// DISASM: s_mov_b32 [[SCR3:s[0-9]+]], m0
// DISASM-NEXT: s_pack_hh_b32_b16 m0, 0, m0
// DISASM-NEXT: cluster_load_async_to_lds_b8 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}] offset:-64 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// DISASM-NEXT: s_mov_b32 m0, [[SCR3]]
// DISASM-NEXT: s_branch
// DISASM: s_mov_b32 [[SCR4:s[0-9]+]], m0
// DISASM-NEXT: s_pack_hh_b32_b16 m0, 0, m0
// DISASM-NEXT: cluster_load_async_to_lds_b64 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}] scale_offset th:TH_LOAD_BYPASS scope:SCOPE_SYS
// DISASM-NEXT: s_mov_b32 m0, [[SCR4]]
// DISASM-NEXT: s_branch
// DISASM: s_mov_b32 [[SCR5:s[0-9]+]], m0
// DISASM-NEXT: s_pack_hh_b32_b16 m0, 0, m0
// DISASM-NEXT: cluster_load_async_to_lds_b128 v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9:]+}}] offset:64
// DISASM-NEXT: s_mov_b32 m0, [[SCR5]]
// DISASM-NEXT: s_branch

// METADATA: .name:           test_cluster_mask
// METADATA: .sgpr_count:     17

// COM: Idempotency: rewriting the output again should produce identical bytes.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// COM: No scratch SGPR means the A0 M0 mask workaround cannot be
// COM: emitted, so the API reports ERROR instead of returning unsafe code.
// RUN: sed -e 's/.amdhsa_next_free_sgpr 16/.amdhsa_next_free_sgpr 106/' \
// RUN:     -e 's/.sgpr_count: 16/.sgpr_count: 106/' %s > %t.highsgpr.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.highsgpr.s -o %t.highsgpr.elf
// RUN: hotswap-rewrite %t.highsgpr.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR \
// RUN:   | %FileCheck --check-prefix=NO-SCRATCH %s
// NO-SCRATCH: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_cluster_mask
.p2align 8
.type test_cluster_mask,@function
test_cluster_mask:
  cluster_load_b64 v[0:1], v2, s[4:5]
  s_wait_loadcnt 0x0
  cluster_load_async_to_lds_b32 v3, v4, s[6:7]
  s_wait_loadcnt 0x0
  cluster_load_b128 v[8:11], v12, s[8:9] offset:64 scale_offset th:TH_LOAD_NT_HT scope:SCOPE_DEV
  s_wait_loadcnt 0x0
  cluster_load_async_to_lds_b8 v13, v14, s[10:11] offset:-64 th:TH_LOAD_NT_HT scope:SCOPE_DEV
  s_wait_loadcnt 0x0
  cluster_load_async_to_lds_b64 v15, v16, s[12:13] scale_offset th:TH_LOAD_BYPASS scope:SCOPE_SYS
  s_wait_loadcnt 0x0
  cluster_load_async_to_lds_b128 v17, v18, s[14:15] offset:64
  s_wait_loadcnt 0x0
  s_endpgm
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
  s_nop 0
.Ltest_cluster_mask_end:
.size test_cluster_mask, .Ltest_cluster_mask_end-test_cluster_mask

.rodata
.p2align 8
.amdhsa_kernel test_cluster_mask
  .amdhsa_next_free_vgpr 20
  .amdhsa_next_free_sgpr 16
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_cluster_mask
      .symbol: test_cluster_mask.kd
      .sgpr_count: 16
      .vgpr_count: 20
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
