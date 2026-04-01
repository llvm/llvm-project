add_aiir_conversion_library(AIIRArithToArmSME
  ArithToArmSME.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ArithToArmSME

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArmSMEDialect
  AIIRArithDialect
  AIIRPass
  AIIRTransforms
  )
