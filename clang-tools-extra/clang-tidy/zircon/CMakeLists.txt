set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyZirconModule
  TemporaryObjectsCheck.cpp
  ZirconTidyModule.cpp

  LINK_LIBS
  clangTidy
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
