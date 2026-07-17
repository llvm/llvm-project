// COM: A dynamic relocation can write outside .text while its addend still
// COM: references displaced code. Direct displacement must decline this input
// COM: and use an appended entry stub so the relocation remains valid.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <relocation_kernel>:
// DISASM-NEXT: s_endpgm
// DISASM-LABEL: <pointed_helper>:
// DISASM-NEXT: s_endpgm
// DISASM: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_get_pc_i64

// RUN: (echo ORIGINAL; %llvm-readelf -Ws %t.elf; \
// RUN:  echo REWRITTEN; %llvm-readelf -Ws %t.out.elf) \
// RUN:  | %FileCheck --check-prefix=SYMBOL %s
// SYMBOL: ORIGINAL
// SYMBOL: [[HELPER:[0-9a-fA-F]+]] {{.*}} pointed_helper
// SYMBOL: REWRITTEN
// SYMBOL: [[HELPER]] {{.*}} pointed_helper

// RUN: (echo ORIGINAL; %llvm-readelf -r %t.elf; \
// RUN:  echo REWRITTEN; %llvm-readelf -r %t.out.elf) \
// RUN:  | %FileCheck --check-prefix=RELOCATION %s
// RELOCATION: ORIGINAL
// RELOCATION: R_AMDGPU_RELATIVE64{{ +}}[[ADDEND:[0-9a-fA-F]+]]
// RELOCATION: REWRITTEN
// RELOCATION: R_AMDGPU_RELATIVE64{{ +}}[[ADDEND]]

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl relocation_kernel
.p2align 8
.type relocation_kernel,@function
relocation_kernel:
  s_endpgm
.Lrelocation_kernel_end:
.size relocation_kernel, .Lrelocation_kernel_end-relocation_kernel

.local pointed_helper
.type pointed_helper,@function
pointed_helper:
  s_endpgm
.Lpointed_helper_end:
.size pointed_helper, .Lpointed_helper_end-pointed_helper

.data
.p2align 3
.globl helper_pointer
.type helper_pointer,@object
helper_pointer:
  .quad pointed_helper
.size helper_pointer, 8

.rodata
.p2align 8
.amdhsa_kernel relocation_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: relocation_kernel
      .symbol: relocation_kernel.kd
      .sgpr_count: 1
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
