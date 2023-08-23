////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
// 
// Copyright (c) 2014-2015, Advanced Micro Devices, Inc. All rights reserved.
// 
// Developed by:
// 
//                 AMD Research and AMD HSA Software Development
// 
//                 Advanced Micro Devices, Inc.
// 
//                 www.amd.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// 
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef AMD_HSA_KERNEL_CODE_H
#define AMD_HSA_KERNEL_CODE_H

#include "amd_hsa_common.h"
#include "hsa.h"

// AMD Kernel Code Version Enumeration Values.
typedef uint32_t amd_kernel_code_version32_t;
enum amd_kernel_code_version_t {
  AMD_KERNEL_CODE_VERSION_MAJOR = 1,
  AMD_KERNEL_CODE_VERSION_MINOR = 1
};

// AMD Machine Kind Enumeration Values.
typedef uint16_t amd_machine_kind16_t;
enum amd_machine_kind_t {
  AMD_MACHINE_KIND_UNDEFINED = 0,
  AMD_MACHINE_KIND_AMDGPU = 1
};

// AMD Machine Version.
typedef uint16_t amd_machine_version16_t;

// AMD Float Round Mode Enumeration Values.
enum amd_float_round_mode_t {
  AMD_FLOAT_ROUND_MODE_NEAREST_EVEN = 0,
  AMD_FLOAT_ROUND_MODE_PLUS_INFINITY = 1,
  AMD_FLOAT_ROUND_MODE_MINUS_INFINITY = 2,
  AMD_FLOAT_ROUND_MODE_ZERO = 3
};

// AMD Float Denorm Mode Enumeration Values.
enum amd_float_denorm_mode_t {
  AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE_OUTPUT = 0,
  AMD_FLOAT_DENORM_MODE_FLUSH_OUTPUT = 1,
  AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE = 2,
  AMD_FLOAT_DENORM_MODE_NO_FLUSH = 3
};

// AMD Compute Program Resource Register One.
typedef uint32_t amd_compute_pgm_rsrc_one32_t;
enum amd_compute_pgm_rsrc_one_t {
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT, 0, 6),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT, 6, 4),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY, 10, 2),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32, 12, 2),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64, 14, 2),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32, 16, 2),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64, 18, 2),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_PRIV, 20, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP, 21, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE, 22, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE, 23, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_BULKY, 24, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER, 25, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1, 26, 6)
};

// AMD System VGPR Workitem ID Enumeration Values.
enum amd_system_vgpr_workitem_id_t {
  AMD_SYSTEM_VGPR_WORKITEM_ID_X = 0,
  AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y = 1,
  AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y_Z = 2,
  AMD_SYSTEM_VGPR_WORKITEM_ID_UNDEFINED = 3
};

// AMD Compute Program Resource Register Two.
typedef uint32_t amd_compute_pgm_rsrc_two32_t;
enum amd_compute_pgm_rsrc_two_t {
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET, 0, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT, 1, 5),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER, 6, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X, 7, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y, 8, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z, 9, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO, 10, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID, 11, 2),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH, 13, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION, 14, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE, 15, 9),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION, 24, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE, 25, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO, 26, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW, 27, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW, 28, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT, 29, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO, 30, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1, 31, 1)
};

// AMD Element Byte Size Enumeration Values.
enum amd_element_byte_size_t {
  AMD_ELEMENT_BYTE_SIZE_2 = 0,
  AMD_ELEMENT_BYTE_SIZE_4 = 1,
  AMD_ELEMENT_BYTE_SIZE_8 = 2,
  AMD_ELEMENT_BYTE_SIZE_16 = 3
};

// AMD Kernel Code Properties.
typedef uint32_t amd_kernel_code_properties32_t;
enum amd_kernel_code_properties_t {
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 0, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR, 1, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR, 2, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 3, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID, 4, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT, 5, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE, 6, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X, 7, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y, 8, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z, 9, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_RESERVED1, 10, 6),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS, 16, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE, 17, 2),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_IS_PTR64, 19, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK, 20, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED, 21, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED, 22, 1),
  AMD_HSA_BITS_CREATE_ENUM_ENTRIES(AMD_KERNEL_CODE_PROPERTIES_RESERVED2, 23, 9)
};

// AMD Power Of Two Enumeration Values.
typedef uint8_t amd_powertwo8_t;
enum amd_powertwo_t {
  AMD_POWERTWO_1 = 0,
  AMD_POWERTWO_2 = 1,
  AMD_POWERTWO_4 = 2,
  AMD_POWERTWO_8 = 3,
  AMD_POWERTWO_16 = 4,
  AMD_POWERTWO_32 = 5,
  AMD_POWERTWO_64 = 6,
  AMD_POWERTWO_128 = 7,
  AMD_POWERTWO_256 = 8
};

// AMD Enabled Control Directive Enumeration Values.
typedef uint64_t amd_enabled_control_directive64_t;
enum amd_enabled_control_directive_t {
  AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_BREAK_EXCEPTIONS = 1,
  AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_DETECT_EXCEPTIONS = 2,
  AMD_ENABLED_CONTROL_DIRECTIVE_MAX_DYNAMIC_GROUP_SIZE = 4,
  AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_GRID_SIZE = 8,
  AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_WORKGROUP_SIZE = 16,
  AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_DIM = 32,
  AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_GRID_SIZE = 64,
  AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_WORKGROUP_SIZE = 128,
  AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRE_NO_PARTIAL_WORKGROUPS = 256
};

// AMD Exception Kind Enumeration Values.
typedef uint16_t amd_exception_kind16_t;
enum amd_exception_kind_t {
  AMD_EXCEPTION_KIND_INVALID_OPERATION = 1,
  AMD_EXCEPTION_KIND_DIVISION_BY_ZERO = 2,
  AMD_EXCEPTION_KIND_OVERFLOW = 4,
  AMD_EXCEPTION_KIND_UNDERFLOW = 8,
  AMD_EXCEPTION_KIND_INEXACT = 16
};

// AMD Control Directives.
#define AMD_CONTROL_DIRECTIVES_ALIGN_BYTES 64
#define AMD_CONTROL_DIRECTIVES_ALIGN __ALIGNED__(AMD_CONTROL_DIRECTIVES_ALIGN_BYTES)
typedef AMD_CONTROL_DIRECTIVES_ALIGN struct amd_control_directives_s {
  amd_enabled_control_directive64_t enabled_control_directives;
  uint16_t enable_break_exceptions;
  uint16_t enable_detect_exceptions;
  uint32_t max_dynamic_group_size;
  uint64_t max_flat_grid_size;
  uint32_t max_flat_workgroup_size;
  uint8_t required_dim;
  uint8_t reserved1[3];
  uint64_t required_grid_size[3];
  uint32_t required_workgroup_size[3];
  uint8_t reserved2[60];
} amd_control_directives_t;

// AMD Kernel Code.
#define AMD_ISA_ALIGN_BYTES 256
#define AMD_KERNEL_CODE_ALIGN_BYTES 64
#define AMD_KERNEL_CODE_ALIGN __ALIGNED__(AMD_KERNEL_CODE_ALIGN_BYTES)
typedef AMD_KERNEL_CODE_ALIGN struct amd_kernel_code_s {
  amd_kernel_code_version32_t amd_kernel_code_version_major;
  amd_kernel_code_version32_t amd_kernel_code_version_minor;
  amd_machine_kind16_t amd_machine_kind;
  amd_machine_version16_t amd_machine_version_major;
  amd_machine_version16_t amd_machine_version_minor;
  amd_machine_version16_t amd_machine_version_stepping;
  int64_t kernel_code_entry_byte_offset;
  int64_t kernel_code_prefetch_byte_offset;
  uint64_t kernel_code_prefetch_byte_size;
  uint64_t max_scratch_backing_memory_byte_size;
  amd_compute_pgm_rsrc_one32_t compute_pgm_rsrc1;
  amd_compute_pgm_rsrc_two32_t compute_pgm_rsrc2;
  amd_kernel_code_properties32_t kernel_code_properties;
  uint32_t workitem_private_segment_byte_size;
  uint32_t workgroup_group_segment_byte_size;
  uint32_t gds_segment_byte_size;
  uint64_t kernarg_segment_byte_size;
  uint32_t workgroup_fbarrier_count;
  uint16_t wavefront_sgpr_count;
  uint16_t workitem_vgpr_count;
  uint16_t reserved_vgpr_first;
  uint16_t reserved_vgpr_count;
  uint16_t reserved_sgpr_first;
  uint16_t reserved_sgpr_count;
  uint16_t debug_wavefront_private_segment_offset_sgpr;
  uint16_t debug_private_segment_buffer_sgpr;
  amd_powertwo8_t kernarg_segment_alignment;
  amd_powertwo8_t group_segment_alignment;
  amd_powertwo8_t private_segment_alignment;
  amd_powertwo8_t wavefront_size;
  int32_t call_convention;
  uint8_t reserved1[12];
  uint64_t runtime_loader_kernel_symbol;
  amd_control_directives_t control_directives;
} amd_kernel_code_t;

// TODO: this struct should be completely gone once debugger designs/implements
// Debugger APIs.
typedef struct amd_runtime_loader_debug_info_s {
  const void* elf_raw;
  size_t elf_size;
  const char *kernel_name;
  const void *owning_segment;
} amd_runtime_loader_debug_info_t;

#endif // AMD_HSA_KERNEL_CODE_H
