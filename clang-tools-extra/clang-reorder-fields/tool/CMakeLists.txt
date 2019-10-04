add_clang_tool(clang-reorder-fields
  ClangReorderFields.cpp
  )

clang_target_link_libraries(clang-reorder-fields
  PRIVATE
  clangBasic
  clangFrontend
  clangRewrite
  clangSerialization
  clangTooling
  clangToolingCore
  )
target_link_libraries(clang-reorder-fields
  PRIVATE
  clangReorderFields
  )
