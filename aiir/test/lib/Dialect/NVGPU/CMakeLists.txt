# Exclude tests from libAIIR.so
add_aiir_library(AIIRNVGPUTestPasses
  TestNVGPUTransforms.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRNVGPUTestPasses PUBLIC
  AIIRIR
  AIIRAffineDialect
  AIIRAnalysis
  AIIRFuncDialect
  AIIRGPUDialect
  AIIRLLVMDialect
  AIIRMemRefDialect
  AIIRNVGPUDialect
  AIIRNVGPUTransforms
  AIIRPass
  AIIRSCFDialect
  AIIRTransformUtils
  )
  
