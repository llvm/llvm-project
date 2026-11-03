add_distinct_clang_unittest(OffloadArchTests
  OffloadArchTest.cpp
  LINK_LIBS
    clangOffloadArch
    LLVMTestingSupport
  LLVM_COMPONENTS
    Support
  )
