set(LLVM_LINK_COMPONENTS
  Core
  Option
  Support
  TargetParser
  )

add_clang_library(clangDependencyScanning
  DependencyScanningFilesystem.cpp
  DependencyScanningService.cpp
  DependencyScanningWorker.cpp
  DependencyScanningUtils.cpp
  DependencyScannerImpl.cpp
  InProcessModuleCache.cpp
  ModuleDepCollector.cpp

  DEPENDS
  ClangDriverOptions

  LINK_LIBS
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  ${LLVM_PTHREAD_LIB}
  )
