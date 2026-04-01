add_aiir_library(AIIRTranslateLib
  AiirTranslateMain.cpp
  Translation.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Tools/aiir-translate

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRParser
  )
