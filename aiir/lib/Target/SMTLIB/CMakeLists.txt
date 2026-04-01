add_aiir_translation_library(AIIRExportSMTLIB
  ExportSMTLIB.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRFuncDialect
  AIIRIR
  AIIRSMT
  AIIRSupport
  AIIRTranslateLib
)
