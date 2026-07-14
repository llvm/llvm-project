// COM: Test HotSwap in-place cluster_load -> global_load patches and verify
// COM: that valid s_clause instructions are preserved.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | \
// RUN:   %FileCheck --check-prefix=METADATA %s

// COM: cluster_load mnemonics should be gone, replaced by global_load
// DISASM-NOT: cluster_load_b32
// DISASM-NOT: cluster_load_b128

// COM: Replacement global_load instructions should be present
// DISASM-DAG: global_load_b32 v0
// DISASM-DAG: global_load_b128 v[4:7]

// COM: Native A0 code generation preserves valid clauses.
// DISASM: s_clause 0x1

// COM: Original global_load instructions should still be there
// DISASM-DAG: global_load_b32 v10
// DISASM-DAG: global_load_b32 v11

// COM: Every rewritten gfx1250 kernel must be labeled for A0.
// METADATA-NOT: .gfx1250_revision: B0
// METADATA: .gfx1250_revision: A0

// COM: Idempotency: rewriting the patched output again should produce
// COM: identical bytes.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_inplace_kernel
.p2align 8
.type test_inplace_kernel,@function
test_inplace_kernel:
  cluster_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0x0
  cluster_load_b128 v[4:7], v[8:9], off
  s_wait_loadcnt 0x0
  s_clause 0x1
  global_load_b32 v10, v[2:3], off
  global_load_b32 v11, v[2:3], off offset:4
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_inplace_kernel_end:
.size test_inplace_kernel, .Ltest_inplace_kernel_end-test_inplace_kernel

.rodata
.p2align 8
.amdhsa_kernel test_inplace_kernel
  .amdhsa_next_free_vgpr 12
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_inplace_kernel
      .symbol: test_inplace_kernel.kd
      .gfx1250_revision: B0
      .sgpr_count: 2
      .vgpr_count: 12
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
