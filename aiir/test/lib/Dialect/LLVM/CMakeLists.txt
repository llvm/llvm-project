# Exclude tests from libAIIR.so
add_aiir_library(AIIRLLVMTestPasses
  TestLowerToLLVM.cpp
  TestPatterns.cpp

  EXCLUDE_FROM_LIBAIIR
  )
aiir_target_link_libraries(AIIRLLVMTestPasses PUBLIC
  AIIRAffineToStandard
  AIIRFuncDialect
  AIIRFuncToLLVM
  AIIRIndexToLLVM
  AIIRIR
  AIIRLLVMDialect
  AIIRLinalgTransforms
  AIIRMathToLLVM
  AIIRMemRefToLLVM
  AIIRMemRefTransforms
  AIIRPass
  AIIRReconcileUnrealizedCasts
  AIIRSCFToControlFlow
  AIIRTransforms
  AIIRUBToLLVM
  AIIRVectorToLLVMPass
  AIIRVectorToSCF
  )
