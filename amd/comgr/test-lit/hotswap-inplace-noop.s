// COM: A kernel requiring no instruction rewrite still needs its gfx1250
// COM: revision metadata retagged from B0 to A0.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | \
// RUN:   %FileCheck --check-prefix=METADATA %s

// COM: Strict mode is accepted even when no strict rewrite matches.
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --strict-mode --output %t.strict.elf \
// RUN:   | %FileCheck --check-prefix=STRICT %s
// STRICT: RESULT: SUCCESS

// COM: No cluster_load or s_clause -- nothing should be patched
// DISASM-NOT: cluster_load
// DISASM-NOT: s_clause
// DISASM: global_load_b32 v0
// DISASM: s_endpgm

// METADATA-NOT: .gfx1250_revision: B0
// METADATA: .gfx1250_revision: A0

// COM: Idempotency: output should be identical on second rewrite.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_noop_kernel
.p2align 8
.type test_noop_kernel,@function
test_noop_kernel:
  global_load_b32 v0, v[2:3], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_noop_kernel_end:
.size test_noop_kernel, .Ltest_noop_kernel_end-test_noop_kernel

.rodata
.p2align 8
.amdhsa_kernel test_noop_kernel
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_noop_kernel
      .symbol: test_noop_kernel.kd
      .gfx1250_revision: B0
      .sgpr_count: 2
      .vgpr_count: 4
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
