# Exclude tests from libMLIR.so
add_flang_library(FIRTestAnalysis
  TestAliasAnalysis.cpp

  DEPENDS
  FIRDialect
  FIRBuilder
  FIRSupport
  FIRTransforms
  FIRAnalysis
  ${dialect_libs}

  LINK_LIBS
  FIRDialect
  FIRBuilder
  FIRSupport
  FIRTransforms
  FIRAnalysis
  MLIRTestAnalysis

  MLIR_LIBS
  ${dialect_libs}
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRAnalysis
  )

target_include_directories(FIRTestAnalysis
  PRIVATE
  ${MLIR_MAIN_SRC_DIR}/..
  )
