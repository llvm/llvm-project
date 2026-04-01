set(LLVM_LINK_COMPONENTS
  nativecodegen
  native
  orcjit
  support
)

add_aiir_unittest(AIIRExecutionEngineTests
  DynamicMemRef.cpp
  StridedMemRef.cpp
  Invoke.cpp
)

aiir_target_link_libraries(AIIRExecutionEngineTests
  PRIVATE
  AIIRArithToLLVM
  AIIRMemRefToLLVM
  AIIRReconcileUnrealizedCasts
  AIIRRegisterAllDialects
)
target_link_libraries(AIIRExecutionEngineTests
  PRIVATE
  AIIRExecutionEngine
)
