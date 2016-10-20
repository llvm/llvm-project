set(LLVM_LINK_COMPONENTS
  support
  )

add_clang_library(clangChangeNamespace
  ChangeNamespace.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangLex
  clangTooling
  clangToolingCore
  )

add_subdirectory(tool)
