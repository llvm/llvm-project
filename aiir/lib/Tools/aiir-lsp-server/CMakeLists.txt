add_aiir_library(AIIRLspServerLib
  LSPServer.cpp
  AIIRServer.cpp
  AiirLspServerMain.cpp
  Protocol.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Tools/aiir-lsp-server

  LINK_COMPONENTS
  SupportLSP

  LINK_LIBS PUBLIC
  AIIRBytecodeWriter
  AIIRFunctionInterfaces
  AIIRIR
  AIIRLspServerSupportLib
  AIIRParser
  )
