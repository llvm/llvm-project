add_clang_tool(clang-reorder-fields ClangReorderFields.cpp)

target_link_libraries(clang-reorder-fields
  PRIVATE
  clangBasic
  clangFrontend
  clangReorderFields
  clangRewrite
  clangTooling
  clangToolingCore
  )
