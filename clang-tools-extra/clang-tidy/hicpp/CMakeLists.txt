set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyHICPPModule
  ExceptionBaseclassCheck.cpp
  MultiwayPathsCoveredCheck.cpp
  NoAssemblerCheck.cpp
  HICPPTidyModule.cpp
  SignedBitwiseCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTidy
  clangTidyBugproneModule
  clangTidyCppCoreGuidelinesModule
  clangTidyGoogleModule
  clangTidyMiscModule
  clangTidyModernizeModule
  clangTidyPerformanceModule
  clangTidyReadabilityModule
  clangTidyUtils
  )
