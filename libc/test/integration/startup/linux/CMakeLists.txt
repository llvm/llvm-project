if(NOT (EXISTS ${LIBC_SOURCE_DIR}/startup/linux/${LIBC_TARGET_ARCHITECTURE}))
  message("Skipping startup integration tests for target architecture ${LIBC_TARGET_ARCHITECTURE}.")
  return()
endif()

add_custom_target(libc-startup-tests)
add_dependencies(libc-integration-tests libc-startup-tests)

add_integration_test(
  startup_args_test
  SUITE libc-startup-tests
  SRCS
    args_test.cpp
  ARGS
    1 2 3
  ENV
    FRANCE=Paris
    GERMANY=Berlin
)

add_integration_test(
  startup_no_envp_test
  SUITE libc-startup-tests
  SRCS
    main_without_envp.cpp
)

add_integration_test(
  startup_no_args_test
  SUITE libc-startup-tests
  SRCS
    main_without_args.cpp
)

add_integration_test(
  startup_tls_test
  SUITE libc-startup-tests
  SRCS
    tls_test.cpp
  DEPENDS
    libc.include.sys_mman
    libc.src.errno.errno
    libc.src.sys.mman.mmap
)

add_integration_test(
  init_fini_array_test
  SUITE libc-startup-tests
  SRCS
    init_fini_array_test.cpp
)
