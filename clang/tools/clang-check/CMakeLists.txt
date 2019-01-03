set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Option
  Support
  )

add_clang_executable(clang-check
  ClangCheck.cpp
  )

target_link_libraries(clang-check
  PRIVATE
  clangAPINotes
  clangAST
  clangBasic
  clangDriver
  clangFrontend
  clangRewriteFrontend
  clangSerialization
  clangStaticAnalyzerFrontend
  clangTooling
  )

install(TARGETS clang-check
  RUNTIME DESTINATION bin)
