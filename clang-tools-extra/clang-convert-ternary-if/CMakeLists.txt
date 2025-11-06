#===- CMakeLists.txt - clang-convert-ternary-if tool ---------------------===#
# This defines the build configuration for the standalone refactoring tool.

add_clang_executable(clang-convert-ternary-if
  ConvertTernaryIf.cpp
  ToolMain.cpp
)

target_link_libraries(clang-convert-ternary-if
  PRIVATE
  clangTooling
  clangBasic
  clangAST
  clangASTMatchers
  clangRewrite
  clangFrontend
)

install(TARGETS clang-convert-ternary-if RUNTIME DESTINATION bin)

