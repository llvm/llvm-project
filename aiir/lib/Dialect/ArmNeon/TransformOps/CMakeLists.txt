add_aiir_dialect_library(AIIRArmNeonVectorTransformOps
  ArmNeonVectorTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/ArmNeon/TransformOps

  DEPENDS
  AIIRArmNeonVectorTransformOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRVectorDialect
  AIIRTransformDialect
  AIIRArmNeonDialect
  AIIRArmNeonTransforms
  )
