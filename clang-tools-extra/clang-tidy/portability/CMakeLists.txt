set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyPortabilityModule
  PortabilityTidyModule.cpp
  RestrictSystemIncludesCheck.cpp
  SIMDIntrinsicsCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyUtils
  clangTooling
  )
