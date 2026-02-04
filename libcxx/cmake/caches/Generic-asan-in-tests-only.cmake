# Don't build the library itself with ASAN support, but enable ASAN in the test suite.
set(LIBCXX_TEST_PARAMS "use_sanitizer=Address" CACHE STRING "")
set(LIBCXXABI_TEST_PARAMS "${LIBCXX_TEST_PARAMS}" CACHE STRING "")
