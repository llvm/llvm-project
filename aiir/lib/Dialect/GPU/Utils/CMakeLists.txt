add_aiir_dialect_library(AIIRGPUUtils
  Utils.cpp
  DistributionUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/GPU/Utils

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRAffineDialect
  AIIRGPUDialect
  AIIRSupport
  AIIRIR
  )
