set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_tool(pp-trace
  PPTrace.cpp
  PPCallbacksTracker.cpp

  DEPENDS
  ClangDriverOptions
  )

clang_target_link_libraries(pp-trace
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  clangTooling
  )
