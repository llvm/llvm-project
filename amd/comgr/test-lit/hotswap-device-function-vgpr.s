// COM: A scratch-register patch in a device function cannot be charged to its
// COM: caller kernels without reachability information. Keep the original
// COM: instruction until device-function call-graph accounting is implemented.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// LOG: hotswap: cannot verify VGPR capacity for a patch site outside a known kernel because its calling kernels are unknown; declining optional patch.
// LOG: RESULT: SUCCESS

// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// DISASM-LABEL: <device_helper>:
// DISASM-NEXT:  ds_store_addtid_b32
// DISASM-NEXT:  s_set_pc_i64 s[0:1]

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl test_device_function_vgpr
.p2align 8
.type test_device_function_vgpr,@function
test_device_function_vgpr:
  s_call_i64 s[0:1], device_helper
  s_endpgm
.Ltest_device_function_vgpr_end:
.size test_device_function_vgpr, .Ltest_device_function_vgpr_end-test_device_function_vgpr

.p2align 8
.type device_helper,@function
device_helper:
  ds_store_addtid_b32 v2 offset:64
  s_set_pc_i64 s[0:1]
.Ldevice_helper_end:
.size device_helper, .Ldevice_helper_end-device_helper

.rodata
.p2align 8
.amdhsa_kernel test_device_function_vgpr
  .amdhsa_next_free_vgpr 3
  .amdhsa_next_free_sgpr 2
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: test_device_function_vgpr
      .symbol: test_device_function_vgpr.kd
      .sgpr_count: 2
      .vgpr_count: 3
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 64
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
