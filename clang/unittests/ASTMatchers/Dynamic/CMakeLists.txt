set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_unittest(DynamicASTMatchersTests
  VariantValueTest.cpp
  ParserTest.cpp
  RegistryTest.cpp
  )

clang_target_link_libraries(DynamicASTMatchersTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangDynamicASTMatchers
  clangFrontend
  clangSerialization
  clangTesting
  clangTooling
  )
