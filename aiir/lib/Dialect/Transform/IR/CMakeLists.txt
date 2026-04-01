add_aiir_dialect_library(AIIRTransformDialect
  TransformAttrs.cpp
  TransformDialect.cpp
  TransformOps.cpp
  TransformTypes.cpp
  Utils.cpp

  LINK_LIBS PUBLIC
  AIIRCastInterfaces
  AIIRFunctionInterfaces
  AIIRIR
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRLoopLikeInterface
  AIIRParser
  AIIRPass
  AIIRRewrite
  AIIRSideEffectInterfaces
  AIIRTransforms
  AIIRTransformDialectInterfaces
  AIIRTransformDialectUtils
  )
