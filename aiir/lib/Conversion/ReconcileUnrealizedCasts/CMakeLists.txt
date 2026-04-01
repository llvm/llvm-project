add_aiir_conversion_library(AIIRReconcileUnrealizedCasts
  ReconcileUnrealizedCasts.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ReconcileUnrealizedCasts

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRTransforms
  )
