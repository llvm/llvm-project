add_custom_target(libc_include_tests)

add_libc_test(
  sys_queue_test
  SUITE
    libc_include_tests
  SRCS
    sys/queue_test.cpp
  DEPENDS
    libc.include.sys_queue
    libc.src.__support.char_vector
    libc.src.__support.CPP.string
)
