set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(pp-trace
  PPTrace.cpp
  PPCallbacksTracker.cpp
  )

target_link_libraries(pp-trace
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangTooling
  )
