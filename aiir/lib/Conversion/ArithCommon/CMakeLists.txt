add_aiir_conversion_library(AIIRArithAttrToLLVMConversion
  AttrToLLVMConverter.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRLLVMDialect
  )
