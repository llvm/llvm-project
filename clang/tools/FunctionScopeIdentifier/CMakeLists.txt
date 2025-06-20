add_clang_executable(FunctionScopeIdentifier
  FunctionScopeIdentifier.cpp
)

target_link_libraries(FunctionScopeIdentifier
  PRIVATE
  clangTooling
  clangBasic
  clangASTMatchers
)
