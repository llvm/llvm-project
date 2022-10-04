add_custom_target(unistd-integration-tests)
add_dependencies(libc-integration-tests unistd-integration-tests)

add_integration_test(
  fork_test
  SUITE
    unistd-integration-tests
  SRCS
    fork_test.cpp
  LOADER
    libc.loader.linux.crt1
  DEPENDS
    libc.include.errno
    libc.include.signal
    libc.include.sys_wait
    libc.include.unistd
    libc.src.signal.raise
    libc.src.sys.wait.wait
    libc.src.unistd.fork
)
