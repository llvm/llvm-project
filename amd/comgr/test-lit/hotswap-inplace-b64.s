// COM: Test HotSwap in-place patch: cluster_load_b64 -> global_load_b64.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: cluster_load_b64 should be swapped to global_load_b64
// DISASM-NOT: cluster_load_b64
// DISASM-DAG: global_load_b64 v[0:1]

// COM: Idempotency: output should be identical on second rewrite.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_b64_kernel
.p2align 8
.type test_b64_kernel,@function
test_b64_kernel:
  cluster_load_b64 v[0:1], v[2:3], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_b64_kernel_end:
.size test_b64_kernel, .Ltest_b64_kernel_end-test_b64_kernel

.rodata
.p2align 8
.amdhsa_kernel test_b64_kernel
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
