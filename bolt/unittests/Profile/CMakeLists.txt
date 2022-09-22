add_bolt_unittest(ProfileTests
  DataAggregator.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB
  )

target_link_libraries(ProfileTests
  PRIVATE
  LLVMBOLTProfile
  )

