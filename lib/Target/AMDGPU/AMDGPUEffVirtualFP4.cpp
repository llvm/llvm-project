// Efficient Integration patch for AMDGPU backend to support virtual FP4/MXFP4
// This extends the existing SWMMAC infrastructure with efficient FP4/MXFP4 operations

#include "llvm/IR/IntrinsicsVFP4.h"
#include "llvm/Support/EffVirtFp4Hw.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Target/AMDGPU/AMDGPUSubtarget.h"
#include "llvm/Target/AMDGPU/AMDGPUInstrInfo.h"

using namespace llvm;

// Extend AMDGPU target lowering to support efficient virtual FP4/MXFP4 operations
namespace llvm {

class AMDGPUEffVirtualFP4Lowering : public TargetLowering {
public:
  explicit AMDGPUEffVirtualFP4Lowering(const AMDGPUSubtarget &STI)
      : TargetLowering(STI) {
    // Initialize efficient virtual FP4 hardware
    init_efficient_virtual_fp4_hw();
    
    // Set up type conversions for efficient FP4
    addTypeForExtLLTPair(MVT::v1i4, MVT::v1f32);
    addTypeForExtLLTPair(MVT::v2i4, MVT::v2f32);
    addTypeForExtLLTPair(MVT::v4i4, MVT::v4f32);
    addTypeForExtLLTPair(MVT::v8i4, MVT::v8f32);
    addTypeForExtLLTPair(MVT::v16i4, MVT::v16f32);
  }

  // Lower efficient virtual FP4/MXFP4 intrinsics to actual operations
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override {
    switch (Op.getOpcode()) {
      case Intrinsic::fp4_convert_from_f32:
        return lowerEffFp4ConvertFromF32(Op, DAG);
      case Intrinsic::fp4_convert_to_f32:
        return lowerEffFp4ConvertToF32(Op, DAG);
      case Intrinsic::fp4_add:
        return lowerEffFp4Add(Op, DAG);
      case Intrinsic::fp4_mul:
        return lowerEffFp4Mul(Op, DAG);
      case Intrinsic::mxfp4_quantize:
        return lowerEffMxfp4Quantize(Op, DAG);
      case Intrinsic::mxfp4_dequantize:
        return lowerEffMxfp4Dequantize(Op, DAG);
      case Intrinsic::mxfp4_matmul:
        return lowerEffMxfp4Matmul(Op, DAG);
      default:
        return TargetLowering::LowerOperation(Op, DAG);
    }
  }

private:
  // Helper functions to lower efficient virtual instructions
  
  SDValue lowerEffFp4ConvertFromF32(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue Input = Op.getOperand(0);  // FP32 input
    
    // Convert to FP4 representation using efficient integer operations
    // This would be lowered to optimized integer operations
    EVT Vec4i4Ty = EVT::getVectorVT(*DAG.getContext(), MVT::i4, 4);
    
    // Create a node that will be lowered to efficient conversion
    SDValue Ops[] = { Input };
    return DAG.getNode(AMDGPUISD::EFF_FP4_CONVERT_FROM_F32, DL, Vec4i4Ty, Ops);
  }
  
  SDValue lowerEffFp4ConvertToF32(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue Input = Op.getOperand(0);  // FP4 input (as i4 vector)
    
    // Convert FP4 vector to FP32 vector using efficient integer operations
    EVT Vec4f32Ty = EVT::getVectorVT(*DAG.getContext(), MVT::f32, 4);
    
    SDValue Ops[] = { Input };
    return DAG.getNode(AMDGPUISD::EFF_FP4_CONVERT_TO_F32, DL, Vec4f32Ty, Ops);
  }
  
  SDValue lowerEffFp4Add(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue A = Op.getOperand(0);  // FP4 vector A
    SDValue B = Op.getOperand(1);  // FP4 vector B
    
    // Perform efficient FP4 addition using integer math
    // This maps to optimized integer operations that leverage INT4 hardware
    EVT Vec4i4Ty = EVT::getVectorVT(*DAG.getContext(), MVT::i4, 4);
    
    SDValue Ops[] = { A, B };
    return DAG.getNode(AMDGPUISD::EFF_FP4_ADD, DL, Vec4i4Ty, Ops);
  }
  
  SDValue lowerEffFp4Mul(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue A = Op.getOperand(0);  // FP4 vector A
    SDValue B = Op.getOperand(1);  // FP4 vector B
    
    // Perform efficient FP4 multiplication using integer math
    EVT Vec4i4Ty = EVT::getVectorVT(*DAG.getContext(), MVT::i4, 4);
    
    SDValue Ops[] = { A, B };
    return DAG.getNode(AMDGPUISD::EFF_FP4_MUL, DL, Vec4i4Ty, Ops);
  }
  
  SDValue lowerEffMxfp4Quantize(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue Input = Op.getOperand(0);      // FP32 input
    SDValue BlockScale = Op.getOperand(1); // Block scaling factors
    
    // Use efficient MXFP4 quantization using integer operations
    EVT Vec4i4Ty = EVT::getVectorVT(*DAG.getContext(), MVT::i4, 4);
    
    SDValue Ops[] = { Input, BlockScale };
    return DAG.getNode(AMDGPUISD::EFF_MXFP4_QUANTIZE, DL, Vec4i4Ty, Ops);
  }
  
  SDValue lowerEffMxfp4Dequantize(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue Input = Op.getOperand(0);      // MXFP4 input
    SDValue BlockScale = Op.getOperand(1); // Block scaling factors
    
    // Use efficient MXFP4 dequantization using integer operations
    EVT Vec4f32Ty = EVT::getVectorVT(*DAG.getContext(), MVT::f32, 4);
    
    SDValue Ops[] = { Input, BlockScale };
    return DAG.getNode(AMDGPUISD::EFF_MXFP4_DEQUANTIZE, DL, Vec4f32Ty, Ops);
  }
  
  SDValue lowerEffMxfp4Matmul(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue A = Op.getOperand(0);  // Matrix A in MXFP4 format
    SDValue B = Op.getOperand(1);  // Matrix B in MXFP4 format
    SDValue C = Op.getOperand(2);  // Accumulator matrix
    SDValue ScaleA = Op.getOperand(3); // Scale factors for A
    SDValue ScaleB = Op.getOperand(4); // Scale factors for B
    
    // Lower to efficient matrix multiplication using INT4 hardware
    SDValue Ops[] = { A, B, C, ScaleA, ScaleB };
    return DAG.getNode(AMDGPUISD::EFF_MXFP4_MATMUL, DL, C.getValueType(), Ops);
  }
};

} // namespace llvm

// Plugin for AMDGPU target to register efficient virtual FP4 support
#define AMDGPU_EFF_VIRTUAL_FP4_PLUGIN