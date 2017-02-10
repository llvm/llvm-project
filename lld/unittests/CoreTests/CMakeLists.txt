add_lld_unittest(CoreTests
  ParallelTest.cpp
  )

target_link_libraries(CoreTests
  ${LLVM_PTHREAD_LIB}
  )
