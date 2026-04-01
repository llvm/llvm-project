add_aiir_library(AIIRArithAndMathToAPFloatUtils
  Utils.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRVectorDialect
  )

add_aiir_conversion_library(AIIRArithToAPFloat
  ArithToAPFloat.cpp
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ArithToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithAndMathToAPFloatUtils
  AIIRArithDialect
  AIIRArithTransforms
  AIIRFuncDialect
  AIIRFuncUtils
  AIIRVectorDialect
  )

add_aiir_conversion_library(AIIRMathToAPFloat
  MathToAPFloat.cpp
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/MathToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRTransformUtils
  AIIRArithAndMathToAPFloatUtils
  AIIRMathDialect
  AIIRFuncDialect
  AIIRFuncUtils
  AIIRVectorDialect
  )
