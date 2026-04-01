add_aiir_conversion_library(AIIRArmSMEToSCF
  ArmSMEToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ArmSMEToSCF

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArmSMEDialect
  AIIRTransforms
  )
