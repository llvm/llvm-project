add_aiir_conversion_library(AIIRMemRefToEmitC
  MemRefToEmitC.cpp
  MemRefToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/MemRefToEmitC

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIREmitCDialect
  AIIRMemRefDialect
  AIIRTransforms
  )
