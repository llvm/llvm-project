set(LLVM_LINK_COMPONENTS
  BitReader
  BitstreamReader
  Support
  )

add_clang_unittest(SerializationTests
  InMemoryModuleCacheTest.cpp
  ModuleCacheTest.cpp
  )

clang_target_link_libraries(SerializationTests
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangSema
  clangSerialization
  )
