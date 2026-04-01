add_aiir_library(AIIRBytecodeWriter
  BytecodeWriter.cpp
  IRNumbering.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Bytecode

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRSupport
  AIIRBytecodeOpInterface
  )
