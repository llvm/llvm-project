// COM: Production activation, quantization, and shared-expert objects exposed
// COM: gateway fragments that can hold the 20-byte SCC-neutral gfx12 set-PC
// COM: sequence but not the former 28-byte SCC save/restore sequence. Keep SCC
// COM: live across this required DS2 rewrite and provide exactly one 20-byte
// COM: gateway. The pool is beyond s_branch range and no island chain exists.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: assigned 1 SCC-neutral forward gateway(s)
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM \
// RUN:   --implicit-check-not=s_add_pc_i64 \
// RUN:   --implicit-check-not=s_add_co_u32 \
// RUN:   --implicit-check-not=s_add_co_ci_u32 %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=META %s

// DISASM-LABEL: <fragmented_gateway>:
// DISASM-NEXT: s_branch
// DISASM-NEXT: s_nop
// DISASM-LABEL: <gateway_barrier>:
// DISASM-NEXT: s_endpgm
// DISASM-NEXT: s_get_pc_i64 s[0:1]
// DISASM-NEXT: s_add_nc_u64 s[0:1], s[0:1],
// DISASM-NEXT: s_set_pc_i64 s[0:1]
// DISASM: ds_load_b32 v0, v2 offset:256
// DISASM-NEXT: ds_load_b32 v1, v2 offset:768
// DISASM-NEXT: s_wait_dscnt 0x0
// DISASM-NEXT: s_get_pc_i64 s[0:1]
// DISASM-NEXT: s_add_nc_u64 s[0:1], s[0:1],
// DISASM-NEXT: s_set_pc_i64 s[0:1]

// META: .name:           fragmented_gateway
// META: .sgpr_count:     4

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl fragmented_gateway
.p2align 8
.type fragmented_gateway,@function
fragmented_gateway:
  ds_load_2addr_stride64_b32 v[0:1], v2 offset0:1 offset1:3
  s_cbranch_scc1 .Lexit
  s_endpgm
.Lexit:
  s_endpgm
.size fragmented_gateway, .-fragmented_gateway

.type gateway_barrier,@function
gateway_barrier:
  s_endpgm
.size gateway_barrier, .-gateway_barrier
.fill 20, 1, 0

.rept 40000
  s_mov_b32 s4, s5
.endr

.rodata
.p2align 8
.amdhsa_kernel fragmented_gateway
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 0
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: fragmented_gateway
      .symbol: fragmented_gateway.kd
      .sgpr_count: 0
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
