set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_library(clangToolingRefactoring
  AtomicChange.cpp

  LINK_LIBS
  clangBasic
  clangToolingCore
  )
