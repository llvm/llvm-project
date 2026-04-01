add_aiir_dialect_library(AIIRArmNeonTransforms
  LowerContractToNeonPatterns.cpp

  DEPENDS
  AIIRArmNeonIncGen

  LINK_LIBS PUBLIC
  AIIRArmNeonDialect
  AIIRFuncDialect
  AIIRVectorDialect
  AIIRIR
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  )
