add_clang_executable(clang-reorder-fields ClangReorderFields.cpp)

target_link_libraries(clang-reorder-fields
  clangBasic
  clangFrontend
  clangReorderFields
  clangRewrite
  clangTooling
  clangToolingCore
  )

install(TARGETS clang-reorder-fields RUNTIME DESTINATION bin)
