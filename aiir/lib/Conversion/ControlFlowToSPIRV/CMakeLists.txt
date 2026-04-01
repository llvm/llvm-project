add_aiir_conversion_library(AIIRControlFlowToSPIRV
  ControlFlowToSPIRV.cpp
  ControlFlowToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SPIRV
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/IR

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRControlFlowDialect
  AIIRPass
  AIIRSPIRVDialect
  AIIRSPIRVConversion
  AIIRSupport
  AIIRTransformUtils
  )
