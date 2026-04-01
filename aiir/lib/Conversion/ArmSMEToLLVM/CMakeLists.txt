add_aiir_conversion_library(AIIRArmSMEToLLVM
  ArmSMEToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/ArmSMEToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRArmSMETransforms
  AIIRArmSMEDialect
  AIIRTransforms
  AIIRLLVMCommonConversion
  AIIRLLVMDialect)
