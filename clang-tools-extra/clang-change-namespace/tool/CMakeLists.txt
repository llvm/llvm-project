include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_tool(clang-change-namespace
  ClangChangeNamespace.cpp
  )
clang_target_link_libraries(clang-change-namespace
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangRewrite
  clangSerialization
  clangTooling
  clangToolingCore
  )
target_link_libraries(clang-change-namespace
  PRIVATE
  clangChangeNamespace
  )
