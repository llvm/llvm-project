set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(BasicTests
  CharInfoTest.cpp
  DiagnosticTest.cpp
  FileManagerTest.cpp
  FixedPointTest.cpp
  MemoryBufferCacheTest.cpp
  SourceManagerTest.cpp
  )

target_link_libraries(BasicTests
  PRIVATE
  clangAST
  clangBasic
  clangLex
  )
