set(LLVM_LINK_COMPONENTS
  lineeditor
  )

add_aiir_library(AIIRQueryLib
  AiirQueryMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Tools/aiir-query

  LINK_LIBS PUBLIC
  AIIRQuery
  )
