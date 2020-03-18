add_libc_unittest(
  utils_test
  SUITE
    libc_string_unittests
  SRCS
    utils_test.cpp
    memcpy_utils_test.cpp
  DEPENDS
    memory_utils
    standalone_cpp
)

target_compile_definitions(
  utils_test
  PRIVATE
  LLVM_LIBC_MEMCPY_MONITOR=memcpy_monitor
)
