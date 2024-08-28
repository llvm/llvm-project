add_custom_target(unistd-integration-tests)
add_dependencies(libc-integration-tests unistd-integration-tests)

add_integration_test(
  getcwd_test
  SUITE
    unistd-integration-tests
  SRCS
    getcwd_test.cpp
  DEPENDS
    libc.src.__support.CPP.string_view
    libc.src.errno.errno
    libc.src.stdlib.getenv
    libc.src.unistd.getcwd
)

add_integration_test(
  fork_test
  SUITE
    unistd-integration-tests
  SRCS
    fork_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.signal
    libc.include.sys_wait
    libc.include.unistd
    libc.src.pthread.pthread_atfork
    libc.src.signal.raise
    libc.src.sys.wait.wait
    libc.src.sys.wait.wait4
    libc.src.sys.wait.waitpid
    libc.src.unistd.fork
    libc.src.unistd.gettid
    libc.src.__support.OSUtil.osutil
    libc.src.stdlib.exit
    libc.include.sys_syscall
)

if((${LIBC_TARGET_OS} STREQUAL "linux") AND (${LIBC_TARGET_ARCHITECTURE_IS_X86}))
  add_integration_test(
    stack_smashing_test
    SUITE
      unistd-integration-tests
    SRCS
      stack_smashing_test.cpp
    DEPENDS
      libc.include.errno
      libc.include.signal
      libc.include.sys_wait
      libc.include.unistd
      libc.src.compiler.__stack_chk_fail
      libc.src.pthread.pthread_atfork
      libc.src.signal.raise
      libc.src.sys.wait.wait
      libc.src.sys.wait.wait4
      libc.src.sys.wait.waitpid
      libc.src.unistd.fork
    COMPILE_OPTIONS
      -fstack-protector-all
  )
endif()

add_executable(
  libc_execv_test_normal_exit
  EXCLUDE_FROM_ALL
  execv_test_normal_exit.cpp
)
set_target_properties(
  libc_execv_test_normal_exit
  PROPERTIES
    OUTPUT_NAME libc_execv_test_normal_exit
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(
  libc_execv_test_signal_exit
  EXCLUDE_FROM_ALL
  execv_test_signal_exit.cpp
)
set_target_properties(
  libc_execv_test_signal_exit
  PROPERTIES
    OUTPUT_NAME libc_execv_test_signal_exit
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_integration_test(
  execv_test
  SUITE
    unistd-integration-tests
  SRCS
    execv_test.cpp
  DEPENDS
    libc_execv_test_normal_exit
    libc_execv_test_signal_exit
    libc.include.errno
    libc.src.sys.wait.waitpid
    libc.src.unistd.execv
    libc.src.unistd.fork
  ENV
    EXECV_TEST=PASS
)

add_integration_test(
  execve_test
  SUITE
    unistd-integration-tests
  SRCS
    execve_test.cpp
  DEPENDS
    libc_execv_test_normal_exit
    libc_execv_test_signal_exit
    libc.include.errno
    libc.src.sys.wait.waitpid
    libc.src.unistd.execve
    libc.src.unistd.fork
  ENV
    EXECV_TEST=PASS
)
