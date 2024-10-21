set(LLVM_LINK_COMPONENTS
    FrontendOpenMP
    Support
)

add_clang_library(clangTidyQtModule
  QtTidyModule.cpp

  LINK_LIBS
  clangTidy
  clangTidyModernizeModule
  clangTidyUtils

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyQtModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTooling
  )
