set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangTidyHICPPModule
  ExceptionBaseclassCheck.cpp
  HICPPTidyModule.cpp
  MultiwayPathsCoveredCheck.cpp
  NoAssemblerCheck.cpp
  SignedBitwiseCheck.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangSerialization
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
