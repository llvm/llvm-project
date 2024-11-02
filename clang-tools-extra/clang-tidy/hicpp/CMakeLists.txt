set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyHICPPModule
  ExceptionBaseclassCheck.cpp
  HICPPTidyModule.cpp
  IgnoredRemoveResultCheck.cpp
  MultiwayPathsCoveredCheck.cpp
  NoAssemblerCheck.cpp
  SignedBitwiseCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyBugproneModule
  clangTidyCppCoreGuidelinesModule
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

clang_target_link_libraries(clangTidyHICPPModule
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangSerialization
  )
