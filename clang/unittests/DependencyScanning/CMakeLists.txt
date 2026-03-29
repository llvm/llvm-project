add_clang_unittest(ClangDependencyScanningTests
  DependencyScanningFilesystemTest.cpp
  CLANG_LIBS
  clangDependencyScanning
  clangFrontend # For TextDiagnosticPrinter.
  LLVM_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Option
  Support
  )
