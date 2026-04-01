add_aiir_dialect_library(AIIRXeGPUDialect
  XeGPUDialect.cpp
  XeGPUOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/aiir/Dialect/XeGPU

  DEPENDS
  AIIRXeGPUIncGen
  AIIRXeGPUAttrInterfaceIncGen
  AIIRXeGPUAttrsIncGen
  AIIRXeGPUEnumsIncGen

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRIndexDialect
  AIIRAffineUtils
  AIIRArithUtils
  AIIRDialectUtils
  AIIRGPUDialect
  AIIRXeVMDialect
  AIIRIR
  AIIRViewLikeInterface
  AIIRVectorDialect
)
