add_libc_unittest(
  utils_test
  SUITE
    libc_string_unittests
  SRCS
    utils_test.cpp
    memcpy_utils_test.cpp
  DEPENDS
    libc.src.string.memory_utils.memory_utils
    libc.utils.CPP.standalone_cpp
)

target_compile_definitions(
  libc.test.src.string.memory_utils.utils_test
  PRIVATE
  LLVM_LIBC_MEMCPY_MONITOR=memcpy_monitor
)
