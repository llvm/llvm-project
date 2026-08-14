add_llvm_executable(inter-translate inter-translate.cpp)
target_link_libraries(inter-translate PRIVATE
  InterXeMachine
  InterEmit
  MLIRFuncDialect
  MLIRParser
  MLIRTargetLLVMIRImport
  MLIRDLTIDialect
  MLIRLLVMDialect
  MLIRIR
  MLIRSupport
  LLVMAsmParser
  LLVMCore
  LLVMSupport
)
