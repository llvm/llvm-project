add_aiir_conversion_library(AIIRSCFToControlFlow
  SCFToControlFlow.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/SCFToControlFlow

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRControlFlowDialect
  AIIRLLVMDialect
  AIIRSCFDialect
  AIIRSCFTransforms
  AIIRTransforms
  )
