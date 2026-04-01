add_subdirectory(Reader)
add_subdirectory(Writer)

add_aiir_library(AIIRBytecodeOpInterface
  BytecodeOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Bytecode

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRSupport
  )
