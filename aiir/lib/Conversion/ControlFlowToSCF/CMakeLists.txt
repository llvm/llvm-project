add_aiir_conversion_library(AIIRControlFlowToSCF
  ControlFlowToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ControlFlowToSCF

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRAnalysis
  AIIRArithDialect
  AIIRControlFlowDialect
  AIIRFuncDialect
  AIIRSCFDialect
  AIIRUBDialect
  AIIRPass
  AIIRTransformUtils
)
