set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_clang_library(clangDependencyScanning
  DependencyScanningWorker.cpp

  DEPENDS
  ClangDriverOptions

  LINK_LIBS
  clangAST
  clangBasic
  clangDriver
  clangFrontend
  clangFrontendTool
  clangLex
  clangParse
  clangSerialization
  clangTooling
)
