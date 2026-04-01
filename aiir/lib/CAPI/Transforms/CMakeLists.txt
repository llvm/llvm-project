add_aiir_upstream_c_api_library(AIIRCAPITransforms
  Passes.cpp
  Rewrite.cpp

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRTransforms
  AIIRTransformUtils
)
