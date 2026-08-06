add_llvm_executable(inter-opt inter-opt.cpp)
target_link_libraries(inter-opt PRIVATE
  InterXeMachine
  MLIROptLib
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRControlFlowDialect
)
mlir_check_link_libraries(inter-opt)
