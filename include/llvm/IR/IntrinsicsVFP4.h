// FP4 and MXFP4 Intrinsics for LLVM AMDGPU Backend
// Defines the interface between LLVM IR and the virtual FP4/MXFP4 hardware

#ifndef LLVM_IR_INTRINSICS_FP4_H
#define LLVM_IR_INTRINSICS_FP4_H

#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

namespace llvm {

namespace Intrinsic {

// Enum values for FP4 and MXFP4 intrinsics
enum ID {
  // Start after the last AMDGPU intrinsic
  // Note: This is a conceptual definition - actual enum values would need to be properly integrated
  
  // FP4 intrinsics
  fp4_convert_from_f32 = AMDGPU::num_intrinsics,  // Convert from FP32 to FP4
  fp4_convert_to_f32,                           // Convert from FP4 to FP32
  fp4_add,                                      // FP4 addition
  fp4_sub,                                      // FP4 subtraction
  fp4_mul,                                      // FP4 multiplication
  fp4_matmul,                                   // FP4 matrix multiplication
  
  // MXFP4 intrinsics
  mxfp4_quantize,                               // Quantize to MXFP4 with scaling
  mxfp4_dequantize,                             // Dequantize from MXFP4
  mxfp4_matmul,                                 // MXFP4 sparse matrix multiplication
  mxfp4_block_scale,                            // Block scaling operation
  
  num_vfp4_intrinsics
};

}  // namespace Intrinsic

}  // namespace llvm

// Define the intrinsic functions that map to virtual FP4/MXFP4 operations

/*
 * FP4 intrinsic definitions
 */

// Convert FP32 to FP4
// @llvm.vfp4.convert.from.f32(<N x float> %input, float %scale) -> <N x i4>
#define INTRINSIC_VFP4_CONVERT_FROM_F32 "llvm.vfp4.convert.from.f32"

// Convert FP4 to FP32
// @llvm.vfp4.convert.to.f32(<N x i4> %input, float %scale) -> <N x float>
#define INTRINSIC_VFP4_CONVERT_TO_F32 "llvm.vfp4.convert.to.f32"

// FP4 addition
// @llvm.vfp4.add(<N x i4> %a, <N x i4> %b, float %scale) -> <N x i4>
#define INTRINSIC_VFP4_ADD "llvm.vfp4.add"

// FP4 multiplication
// @llvm.vfp4.mul(<N x i4> %a, <N x i4> %b, float %scale) -> <N x i4>
#define INTRINSIC_VFP4_MUL "llvm.vfp4.mul"

/*
 * MXFP4 intrinsic definitions
 */

// Quantize to MXFP4 with block scaling
// @llvm.vmxfp4.quantize(<N x float> %input, <M x i8> %block_scale) -> <N x i4>
#define INTRINSIC_VMXF4_QUANTIZE "llvm.vmxfp4.quantize"

// Dequantize from MXFP4
// @llvm.vmxfp4.dequantize(<N x i4> %input, <M x i8> %block_scale) -> <N x float>
#define INTRINSIC_VMXF4_DEQUANTIZE "llvm.vmxfp4.dequantize"

// MXFP4 sparse matrix multiplication using INT4 hardware
// @llvm.vmxfp4.matmul(<N x i4> %A, <N x i4> %B, <N x i4> %C, 
//                     <M x i8> %scale_a, <M x i8> %scale_b) -> <N x i4>
#define INTRINSIC_VMXF4_MATMUL "llvm.vmxfp4.matmul"

#endif // LLVM_IR_INTRINSICS_FP4_H