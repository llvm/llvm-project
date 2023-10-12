add_clang_unittest(InterpTests
  Descriptor.cpp
  )

clang_target_link_libraries(InterpTests
  PRIVATE
  clangAST
  clangBasic
  clangTooling
  )

  target_link_libraries(InterpTests
  PRIVATE
  clangTesting
)
