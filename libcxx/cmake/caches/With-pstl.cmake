# TODO: Remove this cache file once the PSTL is under `-fexperimental-library`
set(LIBCXX_TEST_PARAMS "std=c++17" CACHE STRING "")
set(LIBCXXABI_TEST_PARAMS "${LIBCXX_TEST_PARAMS}" CACHE STRING "")
set(LIBCXX_ENABLE_PARALLEL_ALGORITHMS ON CACHE BOOL "")
