add_llvm_library(FIRDialect
  FIRAttr.cpp
  FIRDialect.cpp
  FIROps.cpp
  FIRType.cpp
)

add_dependencies(FIRDialect FIROpsIncGen)

target_link_libraries(FIRDialect
  MLIRTargetLLVMIR
  MLIRTargetLLVMIRModuleTranslation
  MLIREDSC
  MLIRExecutionEngine
  MLIRParser
  MLIRSupport
  MLIRStandardToLLVM
  MLIRTransforms
  LLVMAsmParser
  LLVMAsmPrinter
  LLVMRemarks
)

install (TARGETS FIRDialect
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
