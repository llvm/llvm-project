set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

add_clang_library(clangToolingRefactor
  AtomicChange.cpp

  LINK_LIBS
  clangBasic
  clangToolingCore
  )
