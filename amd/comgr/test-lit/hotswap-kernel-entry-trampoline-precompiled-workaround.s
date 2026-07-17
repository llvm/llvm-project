// COM: A `global_wb; v_nop` prologue (llvm/llvm-project#208467) already carries
// COM: the workaround, so HotSwap skips its entry trampoline; a kernel without
// COM: it still gets one.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: Direct displacement creates no appended-stub symbols.
// RUN: %llvm-readelf -s %t.out.elf | %FileCheck --check-prefix=SYMS %s
// SYMS-NOT: plain_kernel.stub

// RUN: %llvm-readelf -s %t.out.elf | %FileCheck --check-prefix=NO-WA-STUB %s
// NO-WA-STUB-NOT: precompiled_wa_kernel.stub

// COM: The pre-fixed kernel keeps its in-place workaround, not a redirect stub.
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <precompiled_wa_kernel>:
// DISASM-NEXT: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_endpgm
// DISASM-LABEL: <plain_kernel>:
// DISASM-NEXT: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_setreg_imm32_b32

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
// Prologue already carries the workaround: expect no trampoline.
.globl precompiled_wa_kernel
.p2align 8
.type precompiled_wa_kernel,@function
precompiled_wa_kernel:
  global_wb
  v_nop
  s_endpgm
.Lprecompiled_wa_kernel_end:
.size precompiled_wa_kernel, .Lprecompiled_wa_kernel_end-precompiled_wa_kernel

// No workaround: expect a trampoline.
.globl plain_kernel
.p2align 8
.type plain_kernel,@function
plain_kernel:
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
  s_endpgm
.Lplain_kernel_end:
.size plain_kernel, .Lplain_kernel_end-plain_kernel

.rodata
.p2align 8
.amdhsa_kernel precompiled_wa_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel plain_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: precompiled_wa_kernel
      .symbol: precompiled_wa_kernel.kd
      .sgpr_count: 8
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: plain_kernel
      .symbol: plain_kernel.kd
      .sgpr_count: 8
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
