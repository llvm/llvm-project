set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(graph-builder GraphBuilder.cpp)

target_link_libraries(graph-builder
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  clangASTMatchers
  )