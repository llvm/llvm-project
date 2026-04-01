add_aiir_conversion_library(AIIRArithToAMDGPU
  ArithToAMDGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ArithToAMDGPU

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRAMDGPUDialect
  AIIRAMDGPUUtils
  AIIRArithDialect
  AIIRArithUtils
  AIIRVectorDialect
  AIIRPass
  AIIRTransforms
  )
