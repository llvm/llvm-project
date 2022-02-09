set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangToolingInclusions
  HeaderIncludes.cpp
  IncludeStyle.cpp
  StandardLibrary.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  clangRewrite
  clangToolingCore
  )
