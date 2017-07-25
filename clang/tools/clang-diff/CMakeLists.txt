set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_executable(clang-diff
  ClangDiff.cpp
  )

target_link_libraries(clang-diff
  clangFrontend
  clangTooling
  clangToolingASTDiff
  )
