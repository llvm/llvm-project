set(LLVM_LINK_COMPONENTS
  BitReader
  BitstreamReader
  FrontendOpenMP
  Support
  )

add_clang_unittest(SerializationTests
  ForceCheckFileInputTest.cpp
  InMemoryModuleCacheTest.cpp
  ModuleCacheTest.cpp
  NoCommentsTest.cpp
  SourceLocationEncodingTest.cpp
  VarDeclConstantInitTest.cpp
  )

clang_target_link_libraries(SerializationTests
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangSema
  clangSerialization
  clangTooling
  clangASTMatchers
  )
