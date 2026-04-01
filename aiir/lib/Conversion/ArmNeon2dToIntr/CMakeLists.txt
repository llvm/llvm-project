add_aiir_conversion_library(AIIRArmNeon2dToIntr
  ArmNeon2dToIntr.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ArmNeon2dToIntr

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRArmNeonDialect
  AIIRPass
  AIIRTransforms
  AIIRIR
  AIIRVectorDialect
  )
