add_aiir_dialect_library(AIIRArmNeonDialect
  ArmNeonDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/ArmNeon

  DEPENDS
  AIIRArmNeonIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRSideEffectInterfaces
  )
