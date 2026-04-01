add_aiir_conversion_library(AIIRArithToEmitC
  ArithToEmitC.cpp
  ArithToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ArithToEmitC

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIREmitCDialect
  AIIREmitCTransforms
  AIIRPass
  AIIRTransformUtils
  )
