add_aiir_conversion_library(AIIRIndexToLLVM
  IndexToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/IndexToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIndexDialect
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  )
