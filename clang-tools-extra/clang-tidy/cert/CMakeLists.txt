set(LLVM_LINK_COMPONENTS
  support
  FrontendOpenMP
  )

add_clang_library(clangTidyCERTModule STATIC
  CERTTidyModule.cpp

  LINK_LIBS
  clangTidy
  clangTidyBugproneModule
  clangTidyConcurrencyModule
  clangTidyGoogleModule
  clangTidyMiscModule
  clangTidyModernizeModule
  clangTidyPerformanceModule
  clangTidyReadabilityModule
  clangTidyUtils

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyCERTModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
