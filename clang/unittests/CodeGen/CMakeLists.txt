set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_clang_unittest(ClangCodeGenTests
  BufferSourceTest.cpp
  CodeGenExternalTest.cpp
  IncrementalProcessingTest.cpp
  TBAAMetadataTest.cpp
  CheckTargetFeaturesTest.cpp
  )

clang_target_link_libraries(ClangCodeGenTests
  PRIVATE
  clangAST
  clangBasic
  clangCodeGen
  clangFrontend
  clangLex
  clangParse
  clangSerialization
  )
