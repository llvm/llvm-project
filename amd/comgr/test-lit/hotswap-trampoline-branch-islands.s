// COM: A far source and trampoline can be connected entirely with short
// COM: branches through safe alignment holes. Each hop preserves SCC and the
// COM: trampoline return uses the accepted SGPR-backed set-PC sequence.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// COM: Exercise the large-object planning path with verbose accounting. This
// COM: fixture contains more than 250 KiB of text and requires a deterministic
// COM: forward branch-island chain.
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.scale.elf 2>&1 | %FileCheck --check-prefix=SCALE %s
// SCALE: hotswap: assigned 1 forward s_branch island chain(s)
// SCALE: hotswap: applied 1 instruction patches
// SCALE: hotswap: growWithTrampolines: appended 1 trampoline

// RUN: hotswap-rewrite %t.scale.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=SCALE-IDEM %s
// SCALE-IDEM: IDEMPOTENT: YES

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM \
// RUN:   --implicit-check-not=s_add_pc_i64 %s

// DISASM-LABEL: <test_branch_islands>:
// DISASM-NEXT: s_delay_alu
// DISASM-NEXT: s_branch
// DISASM-LABEL: <gateway_0>:
// DISASM-NEXT: s_endpgm
// DISASM-NEXT: s_branch
// DISASM-LABEL: <gateway_1>:
// DISASM-NEXT: s_endpgm
// DISASM-NEXT: s_branch
// DISASM: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_get_pc_i64 s[14:15]
// DISASM-NEXT: s_add_nc_u64 s[14:15], s[14:15],
// DISASM-NEXT: s_set_pc_i64 s[14:15]

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_branch_islands
.p2align 8
.type test_branch_islands,@function
test_branch_islands:
  s_delay_alu instid0(SALU_CYCLE_1)
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
.size test_branch_islands, .-test_branch_islands

.rept 25000
  s_mov_b32 s0, s1
.endr

.type gateway_0,@function
gateway_0:
  s_endpgm
.size gateway_0, .-gateway_0
.fill 4, 1, 0

.rept 25000
  s_mov_b32 s0, s1
.endr

.type gateway_1,@function
gateway_1:
  s_endpgm
.size gateway_1, .-gateway_1
.fill 4, 1, 0

.rept 15000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_branch_islands
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_branch_islands
      .symbol: test_branch_islands.kd
      .sgpr_count: 14
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
