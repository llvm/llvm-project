add_aiir_dialect_library(AIIROpenACCAnalysis
  OpenACCSupport.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/OpenACC

  LINK_LIBS PUBLIC
  AIIRGPUDialect
  AIIRIR
  AIIROpenACCDialect
  AIIROpenACCUtils
  AIIRSupport
)

