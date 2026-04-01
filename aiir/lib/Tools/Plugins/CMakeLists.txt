add_aiir_library(AIIRPluginsLib
  DialectPlugin.cpp
  PassPlugin.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Tools/Plugins

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRPass
  AIIRSupport
  )
