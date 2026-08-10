set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyLLVMLibcModule STATIC
  CalleeNamespaceCheck.cpp
  ImplementationInNamespaceCheck.cpp
  InlineFunctionDeclCheck.cpp
  LLVMLibcTidyModule.cpp
  RestrictSystemLibcHeadersCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyPortabilityModule
  clangTidyUtils

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyLLVMLibcModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTooling
  )
