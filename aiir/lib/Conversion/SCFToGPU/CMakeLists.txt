add_aiir_conversion_library(AIIRSCFToGPU
  SCFToGPU.cpp
  SCFToGPUPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/SCFToGPU

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRAffineToStandard
  AIIRArithDialect
  AIIRComplexDialect
  AIIRGPUTransforms
  AIIRIR
  AIIRLinalgDialect
  AIIRMemRefDialect
  AIIRPass
  AIIRSupport
  AIIRSideEffectInterfaces
  AIIRTransforms
  )
