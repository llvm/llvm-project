// COM: s_swap_pc_i64 also accepts an absolute immediate target. Translate an
// COM: address inside .text to a text-relative direct target so the second of
// COM: two adjacent far patch sites retains an independently callable entry.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   -Wl,--section-start=.text=0x1000 %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s

// DISASM-LABEL: <test_absolute_call>:
// DISASM-NEXT: s_swap_pc_i64 s[0:1], 0x1010
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop 0
// DISASM-NEXT: s_branch

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_absolute_call
.p2align 8
.type test_absolute_call,@function
test_absolute_call:
  s_swap_pc_i64 s[0:1], 0x1010
  ds_load_2addr_stride64_b64 v[0:3], v4 offset0:1 offset1:2
.Labsolute_target:
  ds_load_2addr_stride64_b64 v[4:7], v8 offset0:3 offset1:4
  s_wait_dscnt 0x0
  s_endpgm
.Ltest_absolute_call_end:
.size test_absolute_call, .Ltest_absolute_call_end-test_absolute_call

// Two separate long sites need two 20-byte SCC-neutral gateways. This
// padding follows s_endpgm and lies outside the function, so it is safe.
.fill 64, 1, 0

// Push the appended trampoline pool beyond s_branch's signed 16-bit dword
// range and force direct-target-aware far-site handling.
.rept 40000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_absolute_call
  .amdhsa_next_free_vgpr 9
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_absolute_call
      .symbol: test_absolute_call.kd
      .sgpr_count: 2
      .vgpr_count: 9
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
