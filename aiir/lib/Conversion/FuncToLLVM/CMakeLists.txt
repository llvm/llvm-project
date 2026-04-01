add_aiir_conversion_library(AIIRFuncToLLVM
  FuncToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/FuncToLLVM

  DEPENDS
  AIIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRAnalysis
  AIIRArithToLLVM
  AIIRControlFlowToLLVM
  AIIRDataLayoutInterfaces
  AIIRFuncDialect
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRTransformUtils
  )
