set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(BasicTests
  CharInfoTest.cpp
  DiagnosticTest.cpp
  FileManagerTest.cpp
  MemoryBufferCacheTest.cpp
  SourceManagerTest.cpp
  VirtualFileSystemTest.cpp
  )

target_link_libraries(BasicTests
  clangBasic
  clangLex
  )
