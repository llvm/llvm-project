add_lld_unittest(CoreTests
  ParallelTest.cpp
  )

target_link_libraries(CoreTests
  ${PTHREAD_LIB}
  )
