// COM: Exercise the large-object SGPR-usage caches with two patch sites in a
// COM: function containing a direct call. The first site builds the per-function
// COM: summary, the call promotes it to the whole-object summary, and the
// COM: second site reuses both cached results. Non-NOP straight-line windows
// COM: let each far site carry its set-PC sequence without a gateway.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: applied 2 instruction patches
// LOG: hotswap: growWithTrampolines: appended 2 trampolines
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM \
// RUN:   --implicit-check-not=s_add_pc_i64 %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=META %s

// DISASM-LABEL: <test_large_usage_cache>:
// DISASM: s_get_pc_i64
// DISASM: s_set_pc_i64
// DISASM: s_call_i64
// DISASM: s_get_pc_i64
// DISASM: s_set_pc_i64
// DISASM: tensor_load_to_lds
// DISASM: tensor_load_to_lds

// META: .name:           test_large_usage_cache
// META: .sgpr_count:     38

// RUN: hotswap-rewrite %t.out.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --check-idempotent | %FileCheck --check-prefix=IDEM %s
// IDEM: IDEMPOTENT: YES

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_large_usage_cache
.p2align 8
.type test_large_usage_cache,@function
test_large_usage_cache:
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s30, s30
  s_mov_b32 s30, s30
  s_mov_b32 s30, s30
  s_mov_b32 s30, s30
  s_mov_b32 s30, s30
  s_mov_b32 s30, s30
  s_call_i64 s[20:21], cache_helper
  tensor_load_to_lds s[0:3], s[4:11]
  s_mov_b32 s31, s31
  s_mov_b32 s31, s31
  s_mov_b32 s31, s31
  s_mov_b32 s31, s31
  s_mov_b32 s31, s31
  s_mov_b32 s31, s31
  s_endpgm
.Ltest_large_usage_cache_end:
.size test_large_usage_cache, .Ltest_large_usage_cache_end-test_large_usage_cache

.type cache_helper,@function
cache_helper:
  s_set_pc_i64 s[20:21]
.Lcache_helper_end:
.size cache_helper, .Lcache_helper_end-cache_helper

// Keep the appended pool beyond short-branch reach without supplying NOP
// padding that could bypass the straight-line expansion path above.
.rept 70000
  s_mov_b32 s0, s1
.endr

.rodata
.p2align 8
.amdhsa_kernel test_large_usage_cache
  .amdhsa_next_free_vgpr 1
  .amdhsa_next_free_sgpr 34
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_large_usage_cache
      .symbol: test_large_usage_cache.kd
      .sgpr_count: 34
      .vgpr_count: 1
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
