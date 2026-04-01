add_aiir_conversion_library(AIIRLinalgToStandard
  LinalgToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/LinalgToStandard

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRFuncDialect
  AIIRIR
  AIIRLinalgDialect
  AIIRLinalgTransforms
  AIIRLLVMDialect
  AIIRMemRefDialect
  AIIRPass
  AIIRSCFDialect
  AIIRTransforms
  )
