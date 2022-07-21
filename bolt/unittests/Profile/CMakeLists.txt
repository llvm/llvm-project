add_bolt_unittest(ProfileTests
  DataAggregator.cpp
  )

target_link_libraries(ProfileTests
  PRIVATE
  LLVMBOLTProfile
  )

