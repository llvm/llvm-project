add_clang_unittest(InterpTests
  Descriptor.cpp
  toAPValue.cpp
  )

clang_target_link_libraries(InterpTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  )

  target_link_libraries(InterpTests
  PRIVATE
  clangTesting
)
