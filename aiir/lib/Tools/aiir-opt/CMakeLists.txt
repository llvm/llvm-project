add_aiir_library(AIIROptLib
  AiirOptMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Tools/aiir-opt

  LINK_LIBS PUBLIC
  AIIRBytecodeWriter
  AIIRDebug
  AIIRObservers
  AIIRPass
  AIIRParser
  AIIRPluginsLib
  AIIRSupport
  AIIRIRDL
  AIIRRemarkStreamer
  )
