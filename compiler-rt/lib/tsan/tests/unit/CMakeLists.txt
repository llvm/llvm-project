set(TSAN_UNIT_TEST_SOURCES
  tsan_clock_test.cc
  tsan_flags_test.cc
  tsan_mman_test.cc
  tsan_mutex_test.cc
  tsan_shadow_test.cc
  tsan_stack_test.cc
  tsan_sync_test.cc
  tsan_unit_test_main.cc)

add_tsan_unittest(TsanUnitTest
  SOURCES ${TSAN_UNIT_TEST_SOURCES})
