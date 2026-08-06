add_llvm_executable(inter-opt inter-opt.cpp)
target_link_libraries(inter-opt PRIVATE
  InterXeMachine
  InterTransforms
  MLIROptLib
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRLLVMDialect
  MLIRDLTIDialect
)
mlir_check_link_libraries(inter-opt)
