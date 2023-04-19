set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  Option
  Support
  TargetParser
  CAS
  )

add_clang_library(clangDependencyScanning
  CASFSActionController.cpp
  DependencyScanningCASFilesystem.cpp
  DependencyScanningFilesystem.cpp
  DependencyScanningService.cpp
  DependencyScanningWorker.cpp
  DependencyScanningTool.cpp
  IncludeTreeActionController.cpp
  ModuleDepCollector.cpp
  ScanAndUpdateArgs.cpp

  DEPENDS
  ClangDriverOptions

  LINK_LIBS
  clangAST
  clangBasic
  clangCAS
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangLex
  clangParse
  clangSerialization
  clangTooling
  )
