add_aiir_dialect_library(AIIRTransformDialectTransforms
  CheckUses.cpp
  InferEffects.cpp
  InterpreterPass.cpp
  PreloadLibraryPass.cpp
  TransformInterpreterUtils.cpp

  DEPENDS
  AIIRTransformDialectTransformsIncGen

  LINK_LIBS PUBLIC
  AIIRFunctionInterfaces
  AIIRTransformDialect
  AIIRIR
  AIIRPass
)
