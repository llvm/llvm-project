set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyFuchsiaModule
  DefaultArgumentsCallsCheck.cpp
  DefaultArgumentsDeclarationsCheck.cpp
  FuchsiaTidyModule.cpp
  MultipleInheritanceCheck.cpp
  OverloadedOperatorCheck.cpp
  StaticallyConstructedObjectsCheck.cpp
  TrailingReturnCheck.cpp
  VirtualInheritanceCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyGoogleModule
  clangTidyUtils

  DEPENDS
  omp_gen
  )

clang_target_link_libraries(clangTidyFuchsiaModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
