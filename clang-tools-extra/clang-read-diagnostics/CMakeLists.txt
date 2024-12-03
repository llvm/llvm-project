add_clang_tool(clang-read-diagnostics
  ClangReadDiagnostics.cpp
  )

clang_target_link_libraries(clang-read-diagnostics
  PRIVATE
  clangFrontend
  )
