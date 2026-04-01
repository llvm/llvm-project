set(LLVM_OPTIONAL_SOURCES
  CanonicalizeGLPass.cpp
  ConvertToReplicatedConstantCompositePass.cpp
  DecorateCompositeTypeLayoutPass.cpp
  LowerABIAttributesPass.cpp
  RewriteInsertsPass.cpp
  SPIRVConversion.cpp
  SPIRVWebGPUTransforms.cpp
  UnifyAliasedResourcePass.cpp
  UpdateVCEPass.cpp
)

add_aiir_dialect_library(AIIRSPIRVConversion
  SPIRVConversion.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SPIRV

  LINK_LIBS PUBLIC
  AIIRArithDialect
  AIIRDialectUtils
  AIIRFuncDialect
  AIIRIR
  AIIRPass
  AIIRSPIRVDialect
  AIIRSupport
  AIIRTransformUtils
  AIIRVectorDialect
  AIIRVectorTransforms
)

add_aiir_dialect_library(AIIRSPIRVTransforms
  CanonicalizeGLPass.cpp
  ConvertToReplicatedConstantCompositePass.cpp
  DecorateCompositeTypeLayoutPass.cpp
  LowerABIAttributesPass.cpp
  RewriteInsertsPass.cpp
  SPIRVWebGPUTransforms.cpp
  UnifyAliasedResourcePass.cpp
  UpdateVCEPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/SPIRV

  DEPENDS
  AIIRSPIRVPassIncGen

  LINK_LIBS PUBLIC
  AIIRPass
  AIIRSPIRVConversion
  AIIRSPIRVUtils
)
