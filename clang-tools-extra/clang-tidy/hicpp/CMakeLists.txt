set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

add_clang_library(clangTidyHICPPModule STATIC
  ExceptionBaseclassCheck.cpp
  HICPPTidyModule.cpp
  MultiwayPathsCoveredCheck.cpp
  SignedBitwiseCheck.cpp

  LINK_LIBS
  clangTidy
  clangTidyBugproneModule
  clangTidyCppCoreGuidelinesModule
  clangTidyGoogleModule
  clangTidyMiscModule
  clangTidyModernizeModule
  clangTidyPerformanceModule
  clangTidyPortabilityModule
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
