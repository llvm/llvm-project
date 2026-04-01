add_subdirectory(Transforms)

add_aiir_dialect_library(AIIRLLVMDialect
  IR/FunctionCallUtils.cpp
  IR/LLVMAttrs.cpp
  IR/LLVMDialect.cpp
  IR/LLVMInterfaces.cpp
  IR/LLVMMemorySlot.cpp
  IR/LLVMTypes.cpp
  IR/LLVMTypeSyntax.cpp
  IR/LLVMDialectBytecode.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/LLVMIR

  DEPENDS
  AIIRLLVMDialectBytecodeIncGen
  AIIRLLVMInterfacesIncGen
  AIIRLLVMIntrinsicOpsIncGen
  AIIRLLVMOpsIncGen
  AIIRLLVMTranslationDialectInterfaceIncGen
  AIIRLLVMTypesIncGen
  AIIROpenMPOpsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  BinaryFormat
  Core

  LINK_LIBS PUBLIC
  AIIRCallInterfaces
  AIIRControlFlowInterfaces
  AIIRDataLayoutInterfaces
  AIIRFunctionInterfaces
  AIIRInferIntRangeInterface
  AIIRInferTypeOpInterface
  AIIRIR
  AIIRMemorySlotInterfaces
  AIIRPtrMemorySpaceInterfaces
  AIIRSideEffectInterfaces
  AIIRSupport
  )

add_aiir_dialect_library(AIIRNVVMDialect
  IR/NVVMDialect.cpp
  IR/BasicPtxBuilderInterface.cpp
  IR/NVVMRequiresSMTraits.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/LLVMIR

  DEPENDS
  AIIRGPUCompilationAttrInterfacesIncGen
  AIIRNVVMOpsIncGen
  AIIRNVVMConversionsIncGen
  AIIRBasicPtxBuilderInterfaceIncGen
  AIIRNVVMRequiresSMTraitsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core
  TargetParser

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRGPUDialect
  AIIROpenMPDialect
  AIIRSideEffectInterfaces
  AIIRInferIntRangeInterface
  )

add_aiir_dialect_library(AIIRROCDLDialect
  IR/ROCDLDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/LLVMIR

  DEPENDS
  AIIRGPUCompilationAttrInterfacesIncGen
  AIIRROCDLOpsIncGen
  AIIRROCDLConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRSideEffectInterfaces
  )

add_aiir_dialect_library(AIIRVCIXDialect
  IR/VCIXDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/LLVMIR

  DEPENDS
  AIIRGPUCompilationAttrInterfacesIncGen
  AIIRVCIXOpsIncGen
  AIIRVCIXConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRSideEffectInterfaces
  )

add_aiir_dialect_library(AIIRXeVMDialect
  IR/XeVMDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/LLVMIR

  DEPENDS
  AIIRGPUCompilationAttrInterfacesIncGen
  AIIRXeVMOpsIncGen
  AIIRXeVMConversionsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core
  TargetParser

  LINK_LIBS PUBLIC
  AIIRDialectUtils
  AIIRIR
  AIIRLLVMDialect
  AIIROpenMPDialect
  AIIRSideEffectInterfaces
)
