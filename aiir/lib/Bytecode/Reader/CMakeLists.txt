add_aiir_library(AIIRBytecodeReader
  BytecodeReader.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Bytecode

  LINK_LIBS PUBLIC
  AIIRAsmParser
  AIIRIR
  AIIRSupport
  )
