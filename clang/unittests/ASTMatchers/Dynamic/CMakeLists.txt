set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(DynamicASTMatchersTests
  VariantValueTest.cpp
  ParserTest.cpp
  RegistryTest.cpp
  )

target_link_libraries(DynamicASTMatchersTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangDynamicASTMatchers
  clangFrontend
  clangTooling
  )
