// Q16-Based Integration patch for AMDGPU backend to support virtual FP4/MXFP4
// This extends the existing SWMMAC infrastructure with Q16-based FP4/MXFP4 operations

#include "llvm/IR/IntrinsicsVFP4.h"
#include "llvm/Support/Q16VirtFp4Hw.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Target/AMDGPU/AMDGPUSubtarget.h"
#include "llvm/Target/AMDGPU/AMDGPUInstrInfo.h"

using namespace llvm;

// Extend AMDGPU target lowering to support Q16-based virtual FP4/MXFP4 operations
namespace llvm {

class AMDGPUQ16VirtualFP4Lowering : public TargetLowering {
public:
  explicit AMDGPUQ16VirtualFP4Lowering(const AMDGPUSubtarget &STI)
      : TargetLowering(STI) {
    // Initialize Q16-based virtual FP4 hardware
    reset_q16_performance_counters();
    
    // Set up type conversions for Q16-based FP4
    addTypeForExtLLTPair(MVT::v1i4, MVT::v1f32);
    addTypeForExtLLTPair(MVT::v2i4, MVT::v2f32);
    addTypeForExtLLTPair(MVT::v4i4, MVT::v4f32);
    addTypeForExtLLTPair(MVT::v8i4, MVT::v8f32);
    addTypeForExtLLTPair(MVT::v16i4, MVT::v16f32);
  }

  // Lower Q16-based virtual FP4/MXFP4 intrinsics to actual operations
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override {
    switch (Op.getOpcode()) {
      case Intrinsic::fp4_convert_from_f32:
        return lowerQ16Fp4ConvertFromF32(Op, DAG);
      case Intrinsic::fp4_convert_to_f32:
        return lowerQ16Fp4ConvertToF32(Op, DAG);
      case Intrinsic::fp4_add:
        return lowerQ16Fp4Add(Op, DAG);
      case Intrinsic::fp4_mul:
        return lowerQ16Fp4Mul(Op, DAG);
      case Intrinsic::mxfp4_quantize:
        return lowerQ16Mxfp4Quantize(Op, DAG);
      case Intrinsic::mxfp4_dequantize:
        return lowerQ16Mxfp4Dequantize(Op, DAG);
      case Intrinsic::mxfp4_matmul:
        return lowerQ16Mxfp4Matmul(Op, DAG);
      default:
        return TargetLowering::LowerOperation(Op, DAG);
    }
  }

private:
  // Helper functions to lower Q16-based virtual instructions
  
  SDValue lowerQ16Fp4ConvertFromF32(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue Input = Op.getOperand(0);  // FP32 input
    
    // Convert to FP4 representation using Q16 fixed-point math
    // This would be lowered to optimized integer operations
    EVT Vec4i4Ty = EVT::getVectorVT(*DAG.getContext(), MVT::i4, 4);
    
    // Create a node that will be lowered to Q16-based conversion
    SDValue Ops[] = { Input };
    return DAG.getNode(AMDGPUISD::Q16_FP4_CONVERT_FROM_F32, DL, Vec4i4Ty, Ops);
  }
  
  SDValue lowerQ16Fp4ConvertToF32(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue Input = Op.getOperand(0);  // FP4 input (as i4 vector)
    
    // Convert FP4 vector to FP32 vector using Q16 fixed-point math
    EVT Vec4f32Ty = EVT::getVectorVT(*DAG.getContext(), MVT::f32, 4);
    
    SDValue Ops[] = { Input };
    return DAG.getNode(AMDGPUISD::Q16_FP4_CONVERT_TO_F32, DL, Vec4f32Ty, Ops);
  }
  
  SDValue lowerQ16Fp4Add(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue A = Op.getOperand(0);  // FP4 vector A
    SDValue B = Op.getOperand(1);  // FP4 vector B
    
    // Perform Q16-based FP4 addition using fixed-point math
    // This maps to optimized integer operations that leverage INT4 hardware
    EVT Vec4i4Ty = EVT::getVectorVT(*DAG.getContext(), MVT::i4, 4);
    
    SDValue Ops[] = { A, B };
    return DAG.getNode(AMDGPUISD::Q16_FP4_ADD, DL, Vec4i4Ty, Ops);
  }
  
  SDValue lowerQ16Fp4Mul(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue A = Op.getOperand(0);  // FP4 vector A
    SDValue B = Op.getOperand(1);  // FP4 vector B
    
    // Perform Q16-based FP4 multiplication using fixed-point math
    EVT Vec4i4Ty = EVT::getVectorVT(*DAG.getContext(), MVT::i4, 4);
    
    SDValue Ops[] = { A, B };
    return DAG.getNode(AMDGPUISD::Q16_FP4_MUL, DL, Vec4i4Ty, Ops);
  }
  
  SDValue lowerQ16Mxfp4Quantize(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue Input = Op.getOperand(0);      // FP32 input
    SDValue ScaleFactor = Op.getOperand(1); // Scale factor
    
    // Use Q16-based MXFP4 quantization using fixed-point operations
    EVT Vec4i4Ty = EVT::getVectorVT(*DAG.getContext(), MVT::i4, 4);
    
    SDValue Ops[] = { Input, ScaleFactor };
    return DAG.getNode(AMDGPUISD::Q16_MXFP4_QUANTIZE, DL, Vec4i4Ty, Ops);
  }
  
  SDValue lowerQ16Mxfp4Dequantize(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue Input = Op.getOperand(0);      // MXFP4 input
    SDValue ScaleFactor = Op.getOperand(1); // Scale factor
    
    // Use Q16-based MXFP4 dequantization using fixed-point operations
    EVT Vec4f32Ty = EVT::getVectorVT(*DAG.getContext(), MVT::f32, 4);
    
    SDValue Ops[] = { Input, ScaleFactor };
    return DAG.getNode(AMDGPUISD::Q16_MXFP4_DEQUANTIZE, DL, Vec4f32Ty, Ops);
  }
  
  SDValue lowerQ16Mxfp4Matmul(SDValue Op, SelectionDAG &DAG) const {
    SDLoc DL(Op);
    SDValue A = Op.getOperand(0);  // Matrix A in MXFP4 format
    SDValue B = Op.getOperand(1);  // Matrix B in MXFP4 format
    SDValue C = Op.getOperand(2);  // Accumulator matrix
    SDValue ScaleA = Op.getOperand(3); // Scale factors for A
    SDValue ScaleB = Op.getOperand(4); // Scale factors for B
    
    // Lower to Q16-based matrix multiplication using INT4 hardware
    SDValue Ops[] = { A, B, C, ScaleA, ScaleB };
    return DAG.getNode(AMDGPUISD::Q16_MXFP4_MATMUL, DL, C.getValueType(), Ops);
  }
};

} // namespace llvm

// Plugin for AMDGPU target to register Q16-based virtual FP4 support
#define AMDGPU_Q16_VIRTUAL_FP4_PLUGIN