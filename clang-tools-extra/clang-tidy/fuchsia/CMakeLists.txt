set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyFuchsiaModule STATIC
  DefaultArgumentsCallsCheck.cpp
  DefaultArgumentsDeclarationsCheck.cpp
  FuchsiaTidyModule.cpp
  OverloadedOperatorCheck.cpp
  StaticallyConstructedObjectsCheck.cpp
  TemporaryObjectsCheck.cpp
  TrailingReturnCheck.cpp
  VirtualInheritanceCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyGoogleModule
  clangTidyMiscModule
  clangTidyUtils

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyFuchsiaModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  )
