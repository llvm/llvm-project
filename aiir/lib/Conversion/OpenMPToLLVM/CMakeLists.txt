add_aiir_conversion_library(AIIROpenMPToLLVM
  OpenMPToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/OpenMPToLLVM

  DEPENDS
  AIIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithToLLVM
  AIIRFuncToLLVM
  AIIRIR
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRMemRefToLLVM
  AIIROpenMPDialect
  AIIRTransforms
  )
