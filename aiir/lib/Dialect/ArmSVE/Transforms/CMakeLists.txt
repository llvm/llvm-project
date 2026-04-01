add_aiir_dialect_library(AIIRArmSVETransforms
  LegalizeForLLVMExport.cpp
  LegalizeVectorStorage.cpp
  LowerContractToSVEPatterns.cpp

  DEPENDS
  AIIRArmSVEConversionsIncGen
  AIIRArmSVEPassIncGen

  LINK_LIBS PUBLIC
  AIIRArmSVEDialect
  AIIRFuncDialect
  AIIRVectorDialect
  AIIRIR
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  )
