set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  TargetParser
  )

add_clang_library(clangTidyPortabilityModule
  PortabilityTidyModule.cpp
  RestrictSystemIncludesCheck.cpp
  SIMDIntrinsicsCheck.cpp
  StdAllocatorConstCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyUtils

  DEPENDS
  omp_gen
  ClangDriverOptions
  )

clang_target_link_libraries(clangTidyPortabilityModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTooling
  )
