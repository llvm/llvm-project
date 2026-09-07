# Exclude tests from libMLIR.so
add_flang_library(FIRTestAnalysis
  TestAliasAnalysis.cpp

  DEPENDS
  FIRDialect
  FIRBuilder
  FIRSupport
  FIRTransforms
  FIRAnalysis

  LINK_LIBS
  FIRDialect
  FIRBuilder
  FIRSupport
  FIRTransforms
  FIRAnalysis
  MLIRTestAnalysis

  MLIR_DEPS
  ${FLANG_MLIR_DIALECT_LIBS}

  MLIR_LIBS
  ${FLANG_MLIR_DIALECT_LIBS}
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRAnalysis
  )

target_include_directories(FIRTestAnalysis
  PRIVATE
  ${MLIR_MAIN_SRC_DIR}/..
  )
