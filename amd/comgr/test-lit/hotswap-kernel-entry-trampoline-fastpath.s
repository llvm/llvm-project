// COM: On a pure B0-to-B0 entry-only rewrite HotSwap takes the no-MC fast path:
// COM: it emits each entry stub from a pre-encoded template with a per-kernel
// COM: scratch pair allocated above the kernel's live SGPR count, bumps the
// COM: descriptor SGPR reservation to cover it, and adds no debug-only .stub
// COM: symbols. A prologue that already carries the workaround is skipped.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --entry-trampolines --output %t.fast.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.fast.elf | %FileCheck --check-prefix=DISASM %s

// COM: The redirect is through the descriptor, so the kernel body is not
// COM: rewritten in place.
// DISASM-LABEL: <plain_kernel>:
// DISASM-NEXT: s_setreg_imm32_b32
// DISASM-NEXT: s_endpgm

// COM: A prologue that already has global_wb; v_nop keeps its in-place
// COM: workaround and gets no redirect stub.
// DISASM-LABEL: <precompiled_wa_kernel>:
// DISASM-NEXT: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_endpgm

// COM: The appended stub uses a per-kernel scratch pair allocated just above
// COM: the kernel's live SGPR count (.sgpr_count 8 -> aligned pair s[8:9]),
// COM: matching the MC path's allocation.
// DISASM: s_get_pc_i64 s[8:9]
// DISASM-NEXT: s_add_co_u32 s8
// DISASM-NEXT: s_add_co_ci_u32 s9
// DISASM-NEXT: s_set_pc_i64 s[8:9]

// COM: The fast path adds no .stub symbols by default (the loader adds none;
// COM: they are only a debugging aid).
// RUN: %llvm-readelf -s %t.fast.elf | %FileCheck --check-prefix=NO-SYMS %s
// NO-SYMS-NOT: .stub

// COM: The per-kernel scratch pair s[8:9] sits above the kernel's 8 live SGPRs,
// COM: so the descriptor SGPR reservation is bumped to cover it (8 -> 10),
// COM: exactly like the MC path.
// RUN: %llvm-readelf --notes %t.fast.elf | %FileCheck --check-prefix=SGPR %s
// SGPR: .name:           plain_kernel
// SGPR: .sgpr_count:     10

// COM: Byte-compare idempotency: a second pass recognizes the installed stub
// COM: (and the in-place workaround) and is a no-op.
// RUN: hotswap-rewrite %t.fast.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --entry-trampolines --output %t.fast2.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// RUN: cmp %t.fast.elf %t.fast2.elf

// COM: AMD_COMGR_HOTSWAP_ENTRY_STUB_SYMBOLS=1 re-enables the .stub symbols on
// COM: the fast path (e.g. for rocgdb).
// RUN: env AMD_COMGR_HOTSWAP_ENTRY_STUB_SYMBOLS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --entry-trampolines --output %t.syms.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// RUN: %llvm-readelf -s %t.syms.elf | %FileCheck --check-prefix=SYMS %s
// SYMS: plain_kernel.stub

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
// No workaround: expect a fast-path redirect stub.
.globl plain_kernel
.p2align 8
.type plain_kernel,@function
plain_kernel:
  s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
  s_endpgm
.Lplain_kernel_end:
.size plain_kernel, .Lplain_kernel_end-plain_kernel

// Prologue already carries the workaround: expect no stub.
.globl precompiled_wa_kernel
.p2align 8
.type precompiled_wa_kernel,@function
precompiled_wa_kernel:
  global_wb
  v_nop
  s_endpgm
.Lprecompiled_wa_kernel_end:
.size precompiled_wa_kernel, .Lprecompiled_wa_kernel_end-precompiled_wa_kernel

.rodata
.p2align 8
.amdhsa_kernel plain_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdhsa_kernel precompiled_wa_kernel
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
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
.end_amdgpu_metadata
