add_aiir_dialect_library(AIIRTransformLoopExtension
  LoopExtension.cpp
  LoopExtensionOps.cpp

  DEPENDS
  AIIRTransformDialectLoopExtensionOpsIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLoopLikeInterface
  AIIRTransformDialect
  AIIRTransforms
)
