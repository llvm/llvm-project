add_aiir_library(AIIRLspServerSupportLib
  CompilationDatabase.cpp
  SourceMgrUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Tools/lsp-server-support

  LINK_COMPONENTS
  SupportLSP

  LINK_LIBS PUBLIC
  AIIRSupport
)
