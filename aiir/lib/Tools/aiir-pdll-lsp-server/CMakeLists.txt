llvm_add_library(AIIRPdllLspServerLib
  LSPServer.cpp
  PDLLServer.cpp
  Protocol.cpp
  AiirPdllLspServerMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Tools/aiir-pdll-lsp-server

  LINK_COMPONENTS
  SupportLSP

  LINK_LIBS PUBLIC
  AIIRPDLLCodeGen
  AIIRPDLLParser
  AIIRLspServerSupportLib
  )
