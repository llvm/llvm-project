add_aiir_dialect_library(AIIRTransformDialectUtils
  DiagnosedSilenceableFailure.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Transform

  LINK_LIBS PUBLIC
  AIIRDialectUtils
  AIIRIR
  AIIRSupport
  AIIRViewLikeInterface
)
