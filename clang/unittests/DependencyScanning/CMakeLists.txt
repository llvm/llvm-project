add_clang_unittest(ClangDependencyScanningTests
  DependencyScanningFilesystemTest.cpp
  InProcessModuleCacheTest.cpp
  CLANG_LIBS
  clangDependencyScanning
  clangFrontend # For TextDiagnosticPrinter.
  LINK_LIBS
  LLVMTestingSupport
  LLVM_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Option
  Support
  )
