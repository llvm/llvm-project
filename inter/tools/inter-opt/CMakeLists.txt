add_llvm_executable(inter-opt inter-opt.cpp)
target_link_libraries(inter-opt PRIVATE
  InterCompiler
  MLIROptLib
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRLLVMDialect
  MLIRDLTIDialect
  MLIRControlFlowToSCF
  MLIRTransformDialectTransforms
  MLIRTransforms
  MLIRUBDialect
  MLIRVectorDialect
  MLIRXeGPUDialect
  MLIRXeVMDialect
)
mlir_check_link_libraries(inter-opt)
