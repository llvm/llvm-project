set(LLVM_LINK_COMPONENTS
  Support
  CAS
  )

add_clang_unittest(ClangCASTests
  CASOptionsTest.cpp
  IncludeTreeTest.cpp
  )

clang_target_link_libraries(ClangCASTests
  PRIVATE
  clangBasic
  clangCAS
  clangDependencyScanning
  )

target_link_libraries(ClangCASTests
  PRIVATE
  LLVMTestingSupport
)
