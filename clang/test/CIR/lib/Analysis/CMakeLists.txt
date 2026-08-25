add_clang_library(CIRTestAnalysis
  TestCIRAliasAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRCIRAnalysis
  MLIRCIR
  MLIRIR
  MLIRAnalysis
  MLIRPass
  MLIRTestAnalysis
)

target_include_directories(CIRTestAnalysis
  PRIVATE
  ${LLVM_MAIN_SRC_DIR}/..
)
