set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(TWClangTests
  DummyTest.cpp
  )

clang_target_link_libraries(TWClangTests
  PRIVATE
  clangBasic
  clangFormat
  clangRewrite
  clangToolingCore
  )
