set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_clang_unittest(ClangCodeGenTests
  BufferSourceTest.cpp
  IncrementalProcessingTest.cpp
  )

target_link_libraries(ClangCodeGenTests
  clangBasic
  clangCodeGen
  clangFrontend
  clangLex
  clangParse
  )
