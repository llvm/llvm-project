add_aiir_conversion_library(AIIRVectorToLLVM
  PARTIAL_SOURCES_INTENDED
  ConvertVectorToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/VectorToLLVM

  DEPENDS
  AIIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRArithAttrToLLVMConversion
  AIIRArithDialect
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRMemRefDialect
  AIIRTargetLLVMIRExport
  AIIRTransforms
  AIIRVectorDialect
  AIIRVectorTransforms
  )

add_aiir_conversion_library(AIIRVectorToLLVMPass
  PARTIAL_SOURCES_INTENDED

  ConvertVectorToLLVMPass.cpp
  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Conversion/VectorToLLVM

  LINK_LIBS PUBLIC
  AIIRVectorToLLVM

  AIIRArmNeonDialect
  AIIRArmNeonTransforms
  AIIRArmSVEDialect
  AIIRArmSVETransforms
  AIIRX86Dialect
  AIIRX86Transforms
)
