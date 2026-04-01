add_aiir_dialect_library(AIIRTransformTuneExtension
  TuneExtension.cpp
  TuneExtensionOps.cpp

  DEPENDS
  AIIRTransformDialectTuneExtensionOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRTransformDialect
)
