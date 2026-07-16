// COM: Verify that hotswap rejects an optional VGPR bump which would make a
// COM: maximum-size workgroup unschedulable, while applying the same patch
// COM: when the kernel has occupancy headroom. Also verify that an applied
// COM: bump updates runtime-visible .vgpr_count metadata.

// RUN: %clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -nostdlib %s -o %t.elf
// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 hotswap-rewrite %t.elf \
// RUN:   amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1250 \
// RUN:   --output %t.out.elf 2>&1 | %FileCheck --check-prefix=LOG %s
// RUN: %llvm-objdump -d %t.out.elf | %FileCheck --check-prefix=DISASM %s
// RUN: %llvm-readelf --notes %t.out.elf | %FileCheck --check-prefix=META %s

// LOG: optional patch for kernel 'capacity_limited' would grow VGPRs from 128 to 129 and reduce capacity to 7 waves/EU, below the 8 waves/EU needed for one maximum-size workgroup; declining patch.
// LOG: RESULT: SUCCESS

// DISASM-LABEL: <capacity_limited>:
// DISASM-NEXT:  ds_store_addtid_b32 v127
// DISASM-LABEL: <has_headroom>:
// DISASM-NOT:   ds_store_addtid_b32
// DISASM:       s_branch
// DISASM:       ds_store_b32 v16, v15

// META: .name:                   capacity_limited
// META: .vgpr_count:             128
// META: .name:                   has_headroom
// META: .vgpr_count:             17

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl capacity_limited
.p2align 8
.type capacity_limited,@function
capacity_limited:
  ds_store_addtid_b32 v127
.irp reg,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31
  v_cmp_eq_u32 vcc_lo, \reg, \reg
.endr
.irp reg,v32,v33,v34,v35,v36,v37,v38,v39,v40,v41,v42,v43,v44,v45,v46,v47,v48,v49,v50,v51,v52,v53,v54,v55,v56,v57,v58,v59,v60,v61,v62,v63
  v_cmp_eq_u32 vcc_lo, \reg, \reg
.endr
.irp reg,v64,v65,v66,v67,v68,v69,v70,v71,v72,v73,v74,v75,v76,v77,v78,v79,v80,v81,v82,v83,v84,v85,v86,v87,v88,v89,v90,v91,v92,v93,v94,v95
  v_cmp_eq_u32 vcc_lo, \reg, \reg
.endr
.irp reg,v96,v97,v98,v99,v100,v101,v102,v103,v104,v105,v106,v107,v108,v109,v110,v111,v112,v113,v114,v115,v116,v117,v118,v119,v120,v121,v122,v123,v124,v125,v126,v127
  v_cmp_eq_u32 vcc_lo, \reg, \reg
.endr
  s_endpgm
.Lcapacity_limited_end:
.size capacity_limited, .Lcapacity_limited_end-capacity_limited

.globl has_headroom
.p2align 8
.type has_headroom,@function
has_headroom:
  ds_store_addtid_b32 v15
.irp reg,v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15
  v_cmp_eq_u32 vcc_lo, \reg, \reg
.endr
  s_endpgm
.Lhas_headroom_end:
.size has_headroom, .Lhas_headroom_end-has_headroom

.rodata
.p2align 8
.amdhsa_kernel capacity_limited
  .amdhsa_next_free_vgpr 128
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
.amdhsa_kernel has_headroom
  .amdhsa_next_free_vgpr 16
  .amdhsa_next_free_sgpr 2
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel

.amdgpu_metadata
  amdhsa.version:
    - 3
    - 0
  amdhsa.kernels:
    - .name: capacity_limited
      .symbol: capacity_limited.kd
      .sgpr_count: 2
      .vgpr_count: 128
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 1024
    - .name: has_headroom
      .symbol: has_headroom.kd
      .sgpr_count: 2
      .vgpr_count: 16
      .kernarg_segment_size: 0
      .group_segment_fixed_size: 0
      .private_segment_fixed_size: 0
      .kernarg_segment_align: 8
      .wavefront_size: 32
      .max_flat_workgroup_size: 256
.end_amdgpu_metadata
