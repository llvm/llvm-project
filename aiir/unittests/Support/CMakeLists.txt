add_aiir_unittest(AIIRSupportTests
  CyclicReplacerCacheTest.cpp
  IndentedOstreamTest.cpp
  StorageUniquerTest.cpp
)

aiir_target_link_libraries(AIIRSupportTests
  PRIVATE AIIRSupport)
