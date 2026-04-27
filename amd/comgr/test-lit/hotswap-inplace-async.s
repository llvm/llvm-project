// COM: Test HotSwap in-place patch: cluster_load_async_to_lds_{b8,b32,b64,b128}
// COM: -> global_load_async_to_lds_{b8,b32,b64,b128}.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// COM: All cluster_load_async_to_lds variants should be replaced
// DISASM-NOT: cluster_load_async_to_lds
// DISASM-DAG: global_load_async_to_lds_b8
// DISASM-DAG: global_load_async_to_lds_b32
// DISASM-DAG: global_load_async_to_lds_b64
// DISASM-DAG: global_load_async_to_lds_b128

// COM: Idempotency: output should be identical on second rewrite.
// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out2.elf \
// RUN:   | %FileCheck --check-prefix=API2 %s
// API2: RESULT: SUCCESS
// RUN: cmp %t.out.elf %t.out2.elf

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_async_kernel
.p2align 8
.type test_async_kernel,@function
test_async_kernel:
  cluster_load_async_to_lds_b8 v1, v[2:3], off
  s_wait_loadcnt 0x0
  cluster_load_async_to_lds_b32 v1, v[2:3], off
  s_wait_loadcnt 0x0
  cluster_load_async_to_lds_b64 v1, v[2:3], off
  s_wait_loadcnt 0x0
  cluster_load_async_to_lds_b128 v1, v[2:3], off
  s_wait_loadcnt 0x0
  s_endpgm
.Ltest_async_kernel_end:
.size test_async_kernel, .Ltest_async_kernel_end-test_async_kernel

.rodata
.p2align 8
.amdhsa_kernel test_async_kernel
  .amdhsa_next_free_vgpr 4
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel
