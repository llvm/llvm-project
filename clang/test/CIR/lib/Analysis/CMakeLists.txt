add_clang_library(CIRTestAnalysis
  TestCIRAliasAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRTestAliasAnalysis
  MLIRTestAnalysis
)

if (NOT CLANG_LINK_CLANG_DYLIB)
  target_link_libraries(CIRTestAnalysis
    PUBLIC
    MLIRCIRAnalysis
    MLIRCIR
    MLIRIR
    MLIRAnalysis
    MLIRPass
  )
endif()

target_include_directories(CIRTestAnalysis
  PRIVATE
  ${LLVM_MAIN_SRC_DIR}/..
)
