add_aiir_conversion_library(AIIRSCFToOpenMP
  SCFToOpenMP.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/SCFToStandard

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRAnalysis
  AIIRArithDialect
  AIIRLLVMDialect
  AIIROpenMPDialect
  AIIRSCFDialect
  AIIRTransforms
  )
