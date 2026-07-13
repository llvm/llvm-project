set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyZirconModule STATIC
  ZirconTidyModule.cpp

  LINK_LIBS
  clangTidy
  clangTidyFuchsiaModule
  clangTidyUtils

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyZirconModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
