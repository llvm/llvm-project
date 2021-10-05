set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  LineEditor
  Option
  OrcJIT
  Support
  )

add_clang_tool(clang-repl
  ClangRepl.cpp
  )

clang_target_link_libraries(clang-repl PUBLIC
  clangBasic
  clangInterpreter
  clangTooling
  )
