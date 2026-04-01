add_aiir_dialect_library(AIIRTransformPDLExtension
  PDLExtension.cpp
  PDLExtensionOps.cpp

  DEPENDS
  AIIRTransformDialectPDLExtensionOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRTransformDialect
  AIIRPDLDialect
  AIIRPDLInterpDialect
)
