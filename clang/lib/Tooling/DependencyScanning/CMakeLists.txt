set(LLVM_LINK_COMPONENTS
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
  DependencyScannerImpl.cpp
  InProcessModuleCache.cpp
  ModuleDepCollector.cpp
  ScanAndUpdateArgs.cpp

  DEPENDS
  ClangDriverOptions

  LINK_LIBS
  clangAST
  clangBasic
  clangCAS
  clangDriver
  clangFrontend
  clangLex
  clangSerialization
  ${LLVM_PTHREAD_LIB}
  )
