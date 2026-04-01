# Exclude tests from libAIIR.so
get_property(conversion_libs GLOBAL PROPERTY AIIR_CONVERSION_LIBS)

set(LIBS
  ${conversion_libs}

  AIIRAffineDialect
  AIIRArithDialect
  AIIRBuiltinToLLVMIRTranslation
  AIIRFuncDialect
  AIIRGPUDialect
  AIIRGPUToGPURuntimeTransforms
  AIIRGPUTransforms
  AIIRIR
  AIIRLLVMToLLVMIRTranslation
  AIIRToLLVMIRTranslationRegistration
  AIIRMemRefDialect
  AIIRMemRefToLLVM
  AIIRNVVMDialect
  AIIRNVVMToLLVMIRTranslation
  AIIRPass
  AIIRROCDLDialect
  AIIRROCDLToLLVMIRTranslation
  AIIRSCFDialect
  AIIRSupport
  AIIRTargetLLVMIRExport
  AIIRTransforms
  AIIRTransformUtils
  AIIRTranslateLib
  AIIRVectorDialect
  AIIRVectorToLLVMPass
  AIIRXeVMDialect
  AIIRXeVMToLLVMIRTranslation
  )

add_aiir_library(AIIRGPUTestPasses
  TestGpuMemoryPromotion.cpp
  TestGpuRewrite.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRGPUTestPasses PUBLIC
  ${LIBS}
  )

