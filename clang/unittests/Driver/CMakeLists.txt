set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  MC
  Option
  Support
  TargetParser
  )

add_clang_unittest(ClangDriverTests
  DistroTest.cpp
  DXCModeTest.cpp
  GCCVersionTest.cpp
  ToolChainTest.cpp
  ModuleCacheTest.cpp
  MultilibBuilderTest.cpp
  MultilibTest.cpp
  SanitizerArgsTest.cpp
  )

clang_target_link_libraries(ClangDriverTests
  PRIVATE
  clangDriver
  clangBasic
  clangFrontend # For TextDiagnosticPrinter.
  clangSerialization
  )
