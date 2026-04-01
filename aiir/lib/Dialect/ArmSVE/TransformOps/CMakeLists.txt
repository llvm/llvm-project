add_aiir_dialect_library(AIIRArmSVEVectorTransformOps
  ArmSVEVectorTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/ArmSVE/TransformOps

  DEPENDS
  AIIRArmSVEVectorTransformOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRVectorDialect
  AIIRTransformDialect
  AIIRArmSVEDialect
  AIIRArmSVETransforms
  )
