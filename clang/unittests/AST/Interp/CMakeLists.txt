add_clang_unittest(InterpTests
  Descriptor.cpp
  )

clang_target_link_libraries(InterpTests
  PRIVATE
  clangAST
  clangBasic
  )

  target_link_libraries(InterpTests
  PRIVATE
  clangTesting
)
