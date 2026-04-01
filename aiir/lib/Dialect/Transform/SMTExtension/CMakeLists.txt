add_aiir_dialect_library(AIIRTransformSMTExtension
  SMTExtension.cpp
  SMTExtensionOps.cpp

  DEPENDS
  AIIRTransformDialectSMTExtensionOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRTransformDialect
  AIIRSMT
)
