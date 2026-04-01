set(LLVM_LINK_COMPONENTS
  nativecodegen
  native
  orcjit
  support
)

# Main API shared library.
add_aiir_upstream_c_api_library(AIIRCAPIExecutionEngine
  ExecutionEngine.cpp

  LINK_LIBS PUBLIC
  AIIRBuiltinToLLVMIRTranslation
  AIIRExecutionEngine
  AIIRLLVMToLLVMIRTranslation
)
