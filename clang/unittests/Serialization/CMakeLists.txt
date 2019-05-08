set(LLVM_LINK_COMPONENTS
  BitReader
  Support
  )

add_clang_unittest(SerializationTests
  InMemoryModuleCacheTest.cpp
  )

target_link_libraries(SerializationTests
  PRIVATE
  clangAST
  clangBasic
  clangLex
  clangSema
  clangSerialization
  )
