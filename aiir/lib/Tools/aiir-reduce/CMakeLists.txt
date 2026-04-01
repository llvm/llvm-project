add_aiir_library(AIIRReduceLib
  AiirReduceMain.cpp

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRParser
  AIIRPass
  AIIRReduce
  AIIRSupport
  )
