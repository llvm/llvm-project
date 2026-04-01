# Exclude tests from libAIIR.so
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
  AIIRTestAnalysis

  AIIR_DEPS
  ${dialect_libs}

  AIIR_LIBS
  ${dialect_libs}
  AIIRFuncDialect
  AIIRLLVMDialect
  AIIRAnalysis
  )

target_include_directories(FIRTestAnalysis
  PRIVATE
  ${AIIR_MAIN_SRC_DIR}/..
  )
