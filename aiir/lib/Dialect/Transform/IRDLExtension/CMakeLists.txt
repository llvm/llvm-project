add_aiir_dialect_library(AIIRTransformDialectIRDLExtension
  IRDLExtension.cpp
  IRDLExtensionOps.cpp

  DEPENDS
  AIIRTransformDialectIRDLExtensionOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRTransformDialect
  AIIRIRDL
)
