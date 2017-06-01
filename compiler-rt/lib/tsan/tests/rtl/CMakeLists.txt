set(TSAN_RTL_TEST_SOURCES
  tsan_bench.cc
  tsan_mop.cc
  tsan_mutex.cc
  tsan_posix.cc
  tsan_string.cc
  tsan_test.cc
  tsan_thread.cc)

if(UNIX)
  list(APPEND TSAN_RTL_TEST_SOURCES tsan_test_util_posix.cc)
endif()

set(TSAN_RTL_TEST_HEADERS
  tsan_test_util.h)

add_tsan_unittest(TsanRtlTest
  SOURCES ${TSAN_RTL_TEST_SOURCES}
  HEADERS ${TSAN_RTL_TEST_HEADERS})
