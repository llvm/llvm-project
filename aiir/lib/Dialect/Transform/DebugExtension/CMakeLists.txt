add_aiir_dialect_library(AIIRTransformDebugExtension
  DebugExtension.cpp
  DebugExtensionOps.cpp

  DEPENDS
  AIIRTransformDialectDebugExtensionOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRTransformDialect
)
