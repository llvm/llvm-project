set(LLVM_LINK_COMPONENTS
  Core
  )

add_clang_unittest(ClangReplInterpreterTests
  IncrementalProcessingTest.cpp
  InterpreterTest.cpp
  )
target_link_libraries(ClangReplInterpreterTests PUBLIC
  clangAST
  clangBasic
  clangInterpreter
  clangFrontend
  )
