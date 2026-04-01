add_aiir_conversion_library(AIIRControlFlowToLLVM
  ControlFlowToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ControlFlowToLLVM

  DEPENDS
  AIIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRAnalysis
  AIIRControlFlowDialect
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRPass
  AIIRTransformUtils
  )
