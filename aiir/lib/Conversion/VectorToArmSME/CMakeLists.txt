add_aiir_conversion_library(AIIRVectorToArmSME
  VectorToArmSME.cpp
  VectorToArmSMEPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/VectorToArmSME

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArmSMEDialect
  AIIRArmSVEDialect
  AIIRLLVMCommonConversion
  )
