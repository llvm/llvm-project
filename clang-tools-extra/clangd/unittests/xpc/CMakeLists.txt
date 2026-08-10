set(LLVM_LINK_COMPONENTS
  support
  )

add_custom_target(ClangdXpcUnitTests)
add_unittest(ClangdXpcUnitTests ClangdXpcTests
  ConversionTests.cpp
  )

target_link_libraries(ClangdXpcTests
  PRIVATE
  clangdXpcJsonConversions
  clangDaemon
  LLVMSupport
  LLVMTestingSupport
  )
