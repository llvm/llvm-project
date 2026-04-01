add_aiir_library(AIIRParser
  Parser.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Parser

  LINK_LIBS PUBLIC
  AIIRAsmParser
  AIIRBytecodeReader
  AIIRIR
  )
