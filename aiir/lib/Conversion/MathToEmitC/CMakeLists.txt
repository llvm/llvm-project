add_aiir_conversion_library(AIIRMathToEmitC
  MathToEmitC.cpp
  MathToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/MathToEmitC

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIREmitCDialect
  AIIRMathDialect
  AIIRPass
  AIIRTransformUtils
)
