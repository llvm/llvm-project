add_aiir_conversion_library(AIIRVectorToXeGPU
  VectorToXeGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/VectorToXeGPU

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRMemRefDialect
  AIIRTransforms
  AIIRVectorDialect
  AIIRVectorToGPU
  AIIRXeGPUDialect
  AIIRXeGPUUtils
  )
