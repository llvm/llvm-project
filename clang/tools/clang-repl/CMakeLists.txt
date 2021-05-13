set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Option
  Support
  )

add_clang_executable(clang-repl
  EXCLUDE_FROM_ALL
  ClangRepl.cpp
  )

target_link_libraries(clang-repl PUBLIC
  clangInterpreter
  clangTooling
  LLVMLineEditor
  )
