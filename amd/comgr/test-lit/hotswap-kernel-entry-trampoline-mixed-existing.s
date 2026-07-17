// COM: An object can contain an existing appended entry stub and a different
// COM: raw kernel entry. Direct displacement would move the existing stub's
// COM: body target without repairing the stub, so the whole rewrite must stay
// COM: on the appended-stub path.

// COM: First create an object where the pre-fixed first kernel is skipped and
// COM: the second kernel receives a fast-path appended stub.
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s \
// RUN:   -o %t.prefixed.elf
// RUN: env AMD_COMGR_HOTSWAP_ENTRY_STUB_SYMBOLS=1 hotswap-rewrite \
// RUN:   %t.prefixed.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   amdgcn-amd-amdhsa--gfx1250:gfx1250-b0-specific+ \
// RUN:   --entry-trampolines --output %t.seed.elf \
// RUN:   | %FileCheck --check-prefix=API %s

// COM: Replace only .text with an equal-sized raw variant. This preserves the
// COM: second descriptor and appended stub while making the first entry need
// COM: the workaround.
// RUN: sed 's/^\.set raw_entry, 0$/.set raw_entry, 1/' \
// RUN:   %s > %t.raw.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %t.raw.s \
// RUN:   -o %t.raw.elf
// RUN: %llvm-objcopy --dump-section .text=%t.raw.text %t.raw.elf
// RUN: %llvm-objcopy --update-section .text=%t.raw.text %t.seed.elf \
// RUN:   %t.mixed.elf

// RUN: hotswap-rewrite %t.mixed.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --entry-trampolines --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: Both original bodies keep their addresses; direct displacement would
// COM: move mixed_second by 16 bytes and leave its old stub target stale.
// RUN: (echo ORIGINAL; %llvm-readelf -Ws %t.mixed.elf; \
// RUN:  echo REWRITTEN; %llvm-readelf -Ws %t.out.elf) \
// RUN:  | %FileCheck --check-prefix=SYMBOL %s
// SYMBOL: ORIGINAL
// SYMBOL-DAG: [[FIRST:[0-9a-fA-F]+]] {{.*}} mixed_first
// SYMBOL-DAG: [[SECOND:[0-9a-fA-F]+]] {{.*}} mixed_second
// SYMBOL: REWRITTEN
// SYMBOL-DAG: [[FIRST]] {{.*}} mixed_first
// SYMBOL-DAG: [[SECOND]] {{.*}} mixed_second
// SYMBOL-DAG: {{[0-9a-fA-F]+}} {{.*}} mixed_first.stub
// SYMBOL-DAG: {{[0-9a-fA-F]+}} {{.*}} mixed_second.stub

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <mixed_first>:
// DISASM-NEXT: s_nop 0
// DISASM-LABEL: <mixed_second>:
// DISASM-NEXT: v_mov_b32_e32 v0, 2
// DISASM: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_get_pc_i64
// DISASM: global_wb
// DISASM-NEXT: v_nop
// DISASM-NEXT: s_get_pc_i64

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.set raw_entry, 0
.text
.globl mixed_first
.p2align 8
.type mixed_first,@function
mixed_first:
.if raw_entry
  s_nop 0
  s_nop 0
  s_nop 0
.else
  global_wb
.endif
  v_nop
  s_endpgm
.Lmixed_first_end:
.size mixed_first, .Lmixed_first_end-mixed_first

.globl mixed_second
.p2align 8
.type mixed_second,@function
mixed_second:
  v_mov_b32_e32 v0, 2
  s_endpgm
.Lmixed_second_end:
.size mixed_second, .Lmixed_second_end-mixed_second

.rodata
.p2align 8
.amdhsa_kernel mixed_first
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.p2align 8
.amdhsa_kernel mixed_second
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: mixed_first
      .symbol: mixed_first.kd
      .sgpr_count: 1
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
    - .name: mixed_second
      .symbol: mixed_second.kd
      .sgpr_count: 1
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
