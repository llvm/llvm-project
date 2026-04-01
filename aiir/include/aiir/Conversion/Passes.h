//===- Passes.h - Conversion Pass Construction and Registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_PASSES_H
#define AIIR_CONVERSION_PASSES_H

#include "aiir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h"
#include "aiir/Conversion/AffineToStandard/AffineToStandard.h"
#include "aiir/Conversion/ArithAndMathToAPFloat/ArithToAPFloat.h"
#include "aiir/Conversion/ArithAndMathToAPFloat/MathToAPFloat.h"
#include "aiir/Conversion/ArithToAMDGPU/ArithToAMDGPU.h"
#include "aiir/Conversion/ArithToArmSME/ArithToArmSME.h"
#include "aiir/Conversion/ArithToEmitC/ArithToEmitCPass.h"
#include "aiir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "aiir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "aiir/Conversion/ArmNeon2dToIntr/ArmNeon2dToIntr.h"
#include "aiir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "aiir/Conversion/ArmSMEToSCF/ArmSMEToSCF.h"
#include "aiir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "aiir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "aiir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "aiir/Conversion/ComplexToLibm/ComplexToLibm.h"
#include "aiir/Conversion/ComplexToROCDLLibraryCalls/ComplexToROCDLLibraryCalls.h"
#include "aiir/Conversion/ComplexToSPIRV/ComplexToSPIRVPass.h"
#include "aiir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "aiir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "aiir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "aiir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h"
#include "aiir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h"
#include "aiir/Conversion/ConvertToEmitC/ConvertToEmitCPass.h"
#include "aiir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "aiir/Conversion/FuncToEmitC/FuncToEmitCPass.h"
#include "aiir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "aiir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h"
#include "aiir/Conversion/GPUCommon/GPUCommonPass.h"
#include "aiir/Conversion/GPUToLLVMSPV/GPUToLLVMSPVPass.h"
#include "aiir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "aiir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "aiir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "aiir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "aiir/Conversion/IndexToSPIRV/IndexToSPIRV.h"
#include "aiir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "aiir/Conversion/MathToEmitC/MathToEmitCPass.h"
#include "aiir/Conversion/MathToFuncs/MathToFuncs.h"
#include "aiir/Conversion/MathToLLVM/MathToLLVM.h"
#include "aiir/Conversion/MathToLibm/MathToLibm.h"
#include "aiir/Conversion/MathToNVVM/MathToNVVM.h"
#include "aiir/Conversion/MathToROCDL/MathToROCDL.h"
#include "aiir/Conversion/MathToSPIRV/MathToSPIRVPass.h"
#include "aiir/Conversion/MathToXeVM/MathToXeVM.h"
#include "aiir/Conversion/MemRefToEmitC/MemRefToEmitCPass.h"
#include "aiir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "aiir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "aiir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "aiir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "aiir/Conversion/OpenACCToSCF/ConvertOpenACCToSCF.h"
#include "aiir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "aiir/Conversion/PDLToPDLInterp/PDLToPDLInterp.h"
#include "aiir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "aiir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "aiir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "aiir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "aiir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "aiir/Conversion/SCFToSPIRV/SCFToSPIRVPass.h"
#include "aiir/Conversion/SPIRVToLLVM/SPIRVToLLVMPass.h"
#include "aiir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "aiir/Conversion/ShardToMPI/ShardToMPI.h"
#include "aiir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "aiir/Conversion/TensorToSPIRV/TensorToSPIRVPass.h"
#include "aiir/Conversion/TosaToArith/TosaToArith.h"
#include "aiir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "aiir/Conversion/TosaToMLProgram/TosaToMLProgram.h"
#include "aiir/Conversion/TosaToSCF/TosaToSCF.h"
#include "aiir/Conversion/TosaToTensor/TosaToTensor.h"
#include "aiir/Conversion/UBToLLVM/UBToLLVM.h"
#include "aiir/Conversion/UBToSPIRV/UBToSPIRV.h"
#include "aiir/Conversion/VectorToAMX/VectorToAMX.h"
#include "aiir/Conversion/VectorToArmSME/VectorToArmSME.h"
#include "aiir/Conversion/VectorToGPU/VectorToGPU.h"
#include "aiir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "aiir/Conversion/VectorToSCF/VectorToSCF.h"
#include "aiir/Conversion/VectorToSPIRV/VectorToSPIRVPass.h"
#include "aiir/Conversion/VectorToXeGPU/VectorToXeGPU.h"
#include "aiir/Conversion/XeGPUToXeVM/XeGPUToXeVM.h"
#include "aiir/Conversion/XeVMToLLVM/XeVMToLLVM.h"

namespace aiir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "aiir/Conversion/Passes.h.inc"

} // namespace aiir

#endif // AIIR_CONVERSION_PASSES_H
