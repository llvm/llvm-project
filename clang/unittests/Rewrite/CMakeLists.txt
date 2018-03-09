set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(RewriteTests
  RewriteBufferTest.cpp
  )
target_link_libraries(RewriteTests
  PRIVATE
  clangRewrite
  )
