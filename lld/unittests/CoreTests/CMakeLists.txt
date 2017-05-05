add_lld_unittest(CoreTests
  ParallelTest.cpp
  )

target_link_libraries(CoreTests
  lldCore ${LLVM_PTHREAD_LIB}
  )
