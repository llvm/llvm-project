set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyMLIRModule STATIC
  MLIRTidyModule.cpp
  OpBuilderCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyReadabilityModule
  clangTidyUtils
  clangTransformer

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyMLIRModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTooling
  )
