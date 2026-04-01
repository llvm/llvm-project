add_aiir_dialect_library(AIIRXeGPUUtils
  XeGPUUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/XeGPU/Utils

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRSCFTransforms
  AIIRGPUDialect
  AIIRXeVMDialect
  AIIRXeGPUDialect
  )
