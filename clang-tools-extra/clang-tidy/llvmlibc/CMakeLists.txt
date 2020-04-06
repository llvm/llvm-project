set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyLLVMLibcModule
  ImplementationInNamespaceCheck.cpp
  LLVMLibcTidyModule.cpp
  RestrictSystemLibcHeadersCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyPortabilityModule
  clangTidyUtils
  clangTooling
  )
