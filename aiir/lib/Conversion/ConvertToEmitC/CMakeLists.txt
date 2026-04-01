add_aiir_conversion_library(AIIRConvertToEmitC
  ConvertToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ConvertToEmitC

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithToEmitC
  AIIRFuncToEmitC
  AIIRMemRefToEmitC
  AIIRPass
  AIIRSCFToEmitC
  AIIRTransformUtils
  )
