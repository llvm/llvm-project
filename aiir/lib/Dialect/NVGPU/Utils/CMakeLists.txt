add_aiir_dialect_library(AIIRNVGPUUtils
  MMAUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Arithmetic/Utils

  LINK_LIBS PUBLIC
  AIIRAffineDialect
  AIIRLLVMDialect
  AIIRNVGPUDialect
  AIIRNVVMDialect
  AIIRVectorDialect
  AIIRIR
  )
