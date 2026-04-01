add_aiir_dialect_library(AIIRSPIRVModuleCombiner
  ModuleCombiner.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SPIRV

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRSPIRVDialect
  AIIRSupport
  )
