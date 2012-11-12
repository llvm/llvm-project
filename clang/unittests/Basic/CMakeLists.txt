add_clang_unittest(BasicTests
  FileManagerTest.cpp
  SourceManagerTest.cpp
  )

target_link_libraries(BasicTests
  clangBasic
  clangLex
  )
