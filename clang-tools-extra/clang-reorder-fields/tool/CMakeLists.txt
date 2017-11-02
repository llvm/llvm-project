add_clang_tool(clang-reorder-fields ClangReorderFields.cpp)

target_link_libraries(clang-reorder-fields
  clangBasic
  clangFrontend
  clangReorderFields
  clangRewrite
  clangTooling
  clangToolingCore
  )
