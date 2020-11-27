set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidySocialpointModule
  DefinitionsInHeadersCheck.cpp
  SocialpointTidyModule.cpp
  SortConstructorInitializersCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyReadabilityModule
  clangTidyUtils

  DEPENDS
  omp_gen
  )

clang_target_link_libraries(clangTidySocialpointModule
  PRIVATE
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangSerialization
  clangTooling
  )
set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )
