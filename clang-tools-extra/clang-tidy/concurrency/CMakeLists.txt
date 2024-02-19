set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyConcurrencyModule
  ConcurrencyTidyModule.cpp
  MtUnsafeCheck.cpp
  ThreadCanceltypeAsynchronousCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyUtils

  DEPENDS
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyConcurrencyModule
  PRIVATE
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangSerialization
  clangTooling
  )
