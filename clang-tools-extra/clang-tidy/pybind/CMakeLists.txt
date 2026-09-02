set(LLVM_LINK_COMPONENTS
  support
  FrontendOpenMP
  )

add_clang_library(clangTidyPybindModule STATIC
  CallGuardInitCheck.cpp
  PybindTidyModule.cpp

  LINK_LIBS
  clangTidy
  clangTidyUtils

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyPybindModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
