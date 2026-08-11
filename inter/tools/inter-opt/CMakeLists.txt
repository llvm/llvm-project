add_llvm_executable(inter-opt inter-opt.cpp)
target_link_libraries(inter-opt PRIVATE
  InterXeMachine
  InterXW
  InterTransforms
  MLIROptLib
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRLLVMDialect
  MLIRDLTIDialect
  MLIRControlFlowToSCF
  MLIRTransformDialectTransforms
  MLIRTransforms
  MLIRUBDialect
)
mlir_check_link_libraries(inter-opt)
