add_aiir_library(AIIRPDLLCodeGen
  CPPGen.cpp
  AIIRGen.cpp

  LINK_LIBS PUBLIC
  AIIRParser
  AIIRPDLLAST
  AIIRPDLDialect
  AIIRSupport
  )
