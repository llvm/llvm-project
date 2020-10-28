set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(BasicTests
  CharInfoTest.cpp
  DiagnosticTest.cpp
  FileManagerTest.cpp
  SourceManagerTest.cpp
  )

clang_target_link_libraries(BasicTests
  PRIVATE
  clangAST
  clangBasic
  clangLex
  )
