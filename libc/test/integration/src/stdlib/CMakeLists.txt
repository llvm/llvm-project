if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()
add_custom_target(stdlib-integration-tests)
add_dependencies(libc-integration-tests stdlib-integration-tests)

add_integration_test(
  getenv_test
  SUITE
    stdlib-integration-tests
  SRCS
    getenv_test.cpp
  DEPENDS
    libc.src.stdlib.getenv
  ENV
    FRANCE=Paris
    GERMANY=Berlin
)

if(${LIBC_TARGET_OS} STREQUAL "linux")
  add_integration_test(
    abort_test
    SUITE
      stdlib-integration-tests
    SRCS
      abort_test.cpp
    DEPENDS
      libc.include.signal
      libc.include.sys_wait
      libc.include.unistd
      libc.src.signal.signal
      libc.src.stdlib._Exit
      libc.src.stdlib.abort
      libc.src.sys.wait.waitpid
      libc.src.unistd.close
      libc.src.unistd.fork
      libc.src.unistd.pipe
      libc.src.unistd.read
      libc.src.unistd.write
  )
endif()
