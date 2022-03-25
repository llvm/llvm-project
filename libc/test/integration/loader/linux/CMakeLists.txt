if(NOT (EXISTS ${LIBC_SOURCE_DIR}/loader/linux/${LIBC_TARGET_ARCHITECTURE}))
  message("Skipping loader integration tests for target architecture ${LIBC_TARGET_ARCHITECTURE}.")
  return()
endif()

add_custom_target(libc-loader-tests)
add_dependencies(libc-integration-tests libc-loader-tests)

add_integration_test(
  loader_args_test
  SUITE libc-loader-tests
  LOADER
    libc.loader.linux.crt1
  SRCS
    args_test.cpp
  ARGS
    1 2 3
  ENV
    FRANCE=Paris
    GERMANY=Berlin
)

add_integration_test(
  loader_no_envp_test
  SUITE libc-loader-tests
  LOADER
    libc.loader.linux.crt1
  SRCS
    main_without_envp.cpp
)

add_integration_test(
  loader_no_args_test
  SUITE libc-loader-tests
  LOADER
    libc.loader.linux.crt1
  SRCS
    main_without_args.cpp
)

if(NOT (${LIBC_TARGET_ARCHITECTURE} STREQUAL "x86_64"))
  return()
endif()

add_integration_test(
  loader_tls_test
  SUITE libc-loader-tests
  LOADER
    libc.loader.linux.crt1
  SRCS
    tls_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.sys_mman
    libc.src.errno.errno
    libc.src.sys.mman.mmap
)
