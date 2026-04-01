add_aiir_dialect_library(AIIRFuncTransformOps
  FuncTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Func/TransformOps

  DEPENDS
  AIIRFuncTransformOpsIncGen

  LINK_LIBS PUBLIC
  AIIRFuncDialect
  AIIRFuncToLLVM
  AIIRIR
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRTransformDialect
  AIIRFuncUtils
)
