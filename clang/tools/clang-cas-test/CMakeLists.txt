set(LLVM_LINK_COMPONENTS
  CAS
  Support
  )

add_clang_tool(clang-cas-test
  ClangCASTest.cpp
  )

clang_target_link_libraries(clang-cas-test PRIVATE
  clangBasic
  clangCAS
  clangFrontend
  )
