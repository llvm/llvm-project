add_aiir_conversion_library(AIIRAsyncToLLVM
  AsyncToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/AsyncToLLVM

  DEPENDS
  AIIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRAsyncDialect
  AIIRFuncToLLVM
  AIIRFuncTransforms
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRTransforms
  )
