add_aiir_conversion_library(AIIRFuncToEmitC
  FuncToEmitC.cpp
  FuncToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/FuncToEmitC

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIREmitCDialect
  AIIRFuncDialect
  AIIRPass
  AIIRTransformUtils
  )
