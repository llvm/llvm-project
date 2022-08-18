set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  MC
  Option
  Support
  )

add_clang_unittest(ClangDriverTests
  DistroTest.cpp
  DXCModeTest.cpp
  ToolChainTest.cpp
  ModuleCacheTest.cpp
  MultilibTest.cpp
  SanitizerArgsTest.cpp
  )

clang_target_link_libraries(ClangDriverTests
  PRIVATE
  clangDriver
  clangBasic
  clangFrontend # For TextDiagnosticPrinter.
  )
