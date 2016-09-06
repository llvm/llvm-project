set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(CFGTests
  CFGTest.cpp
  )

target_link_libraries(CFGTests
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  )
