# Basic tests have few LLVM and Clang dependencies, so linking it as a
# distinct target enables faster iteration times at low cost.
add_distinct_clang_unittest(BasicTests
  CharInfoTest.cpp
  DarwinSDKInfoTest.cpp
  DiagnosticTest.cpp
  FileEntryTest.cpp
  FileManagerTest.cpp
  LineOffsetMappingTest.cpp
  OffloadArchTest.cpp
  SanitizersTest.cpp
  SarifTest.cpp
  SourceManagerTest.cpp
  CLANG_LIBS
  clangBasic
  clangLex
  LINK_LIBS
  LLVMTestingSupport
  LLVM_COMPONENTS
  Support
  )
