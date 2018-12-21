set(TEST_SOURCES
  allocator_test.cc
  buffer_queue_test.cc
  function_call_trie_test.cc
  profile_collector_test.cc
  segmented_array_test.cc
  test_helpers.cc
  xray_unit_test_main.cc)

if (NOT COMPILER_RT_STANDALONE_BUILD OR COMPILER_RT_HAS_LLVMTESTINGSUPPORT)
  list(APPEND TEST_SOURCES
    fdr_controller_test.cc
    fdr_log_writer_test.cc)
endif()

add_xray_unittest(XRayTest SOURCES ${TEST_SOURCES})
