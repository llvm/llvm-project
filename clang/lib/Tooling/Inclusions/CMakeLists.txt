set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangToolingInclusions
  HeaderIncludes.cpp
  IncludeStyle.cpp

  LINK_LIBS
  clangBasic
  clangLex
  clangRewrite
  clangToolingCore
  )
