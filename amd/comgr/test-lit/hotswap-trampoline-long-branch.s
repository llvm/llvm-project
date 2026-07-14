// COM: HSV-009 / PLAT-205406: on gfx1250 A0 a backward s_add_pc_i64 corrupts
// COM: wave state. Required far tensor patches use a forward s_add_pc_i64 and
// COM: an SCC-neutral s_get_pc_i64/s_add_nc_u64/s_set_pc_i64 return instead.
// COM:
// COM: Optional far trampoline rewrites are still allowed to decline;
// COM: hotswap-trampoline-ds-long-branch.s covers that behavior.
// COM: A large .rept filler (~160 KB, non-NOP so it forms no usable sled)
// COM: pushes the pool past s_branch's reach to force the far case.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf

// RUN: hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf \
// RUN:   | %FileCheck --check-prefix=API %s
// API: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=METADATA %s

// DISASM-LABEL: <test_far>:
// DISASM-NEXT: s_mov_b64 vcc, -1
// DISASM-NEXT: s_add_pc_i64
// DISASM-NOT: s_add_pc_i64
// DISASM: s_pack_hh_b32_b16 s4, 0, s4
// DISASM-NEXT: tensor_load_to_lds s[0:3], s[4:11]
// DISASM-NEXT: s_get_pc_i64 s[12:13]
// DISASM-NEXT: s_add_nc_u64 s[12:13]
// DISASM-NEXT: s_set_pc_i64 s[12:13]
// DISASM-NOT: s_add_pc_i64

// METADATA: .name:           test_far
// METADATA: .sgpr_count:     16

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent \
// RUN:   | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

// COM: The far return still needs an aligned SGPR pair. Exhausting all 106
// COM: numbered SGPRs must fail instead of clobbering a program register.
// RUN: sed -e 's/s_mov_b64 vcc, -1/s_mov_b32 s105, 0/' \
// RUN:   -e 's/\.amdhsa_next_free_sgpr 12/.amdhsa_next_free_sgpr 106/' \
// RUN:   -e 's/\.sgpr_count: 14/.sgpr_count: 106/' %s > %t.no-pair.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.no-pair.s -o %t.no-pair.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.no-pair.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=NO-PAIR %s
// NO-PAIR: hotswap: error: safe far return: kernel test_far has no aligned SGPR pair below s106
// NO-PAIR: RESULT: ERROR

// COM: gfx10+ cannot represent an increased SGPR count in the kernel
// COM: descriptor because that field is reserved. A metadata-less object must
// COM: therefore fail rather than emit an under-declared far-return scratch
// COM: allocation or write the reserved descriptor field.
// RUN: sed '/^.amdgpu_metadata$/,/^.end_amdgpu_metadata$/d' %s > %t.nometa.s
// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib \
// RUN:   %t.nometa.s -o %t.nometa.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.nometa.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --expect-status ERROR 2>&1 \
// RUN:   | %FileCheck --check-prefix=NO-METADATA %s
// NO-METADATA: hotswap: error: updateKernelDescriptorSgprCount: kernel 'test_far' requires
// NO-METADATA: descriptor SGPR-count field is reserved
// NO-METADATA: RESULT: ERROR

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_far
.p2align 8
.type test_far,@function
test_far:
  s_mov_b64 vcc, -1
  tensor_load_to_lds s[0:3], s[4:11]
  s_endpgm
  // ~160 KB of non-NOP filler so the appended trampoline pool is beyond
  // s_branch's +-128 KB reach from the tensor_load above (forces the
  // long-branch path).
  .rept 40000
    s_mov_b32 s0, s1
  .endr
.Ltest_far_end:
.size test_far, .Ltest_far_end-test_far

.rodata
.p2align 8
.amdhsa_kernel test_far
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 12
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_far
      .symbol: test_far.kd
      .sgpr_count: 14
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
