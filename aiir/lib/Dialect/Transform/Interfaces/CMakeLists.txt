add_aiir_library(AIIRTransformDialectInterfaces
  MatchInterfaces.cpp
  TransformInterfaces.cpp

  DEPENDS
  AIIRMatchInterfacesIncGen
  AIIRTransformInterfacesIncGen

  LINK_LIBS PUBLIC
  AIIRCastInterfaces
  AIIRIR
  AIIRRewrite
  AIIRSideEffectInterfaces
  AIIRTransforms
  AIIRTransformDialectUtils
)

