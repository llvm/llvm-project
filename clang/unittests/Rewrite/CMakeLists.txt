set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(RewriteTests
  RewriteBufferTest.cpp
  )
clang_target_link_libraries(RewriteTests
  PRIVATE
  clangRewrite
  )
