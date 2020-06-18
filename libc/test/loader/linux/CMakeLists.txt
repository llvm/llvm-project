if(NOT (EXISTS ${LIBC_SOURCE_DIR}/loader/linux/${LIBC_TARGET_MACHINE}))
  message("Skipping loader tests for target machine ${LIBC_TARGET_MACHINE}.")
  return()
endif()

add_loader_test(
  loader_args_test
  SRC
    args_test.cpp
  DEPENDS
    libc.loader.linux.crt1
    libc.src.assert.__assert_fail
    libc.src.signal.raise
    libc.src.stdlib._Exit
    libc.src.stdlib.abort
  ARGS
    1 2 3
  ENV
    FRANCE=Paris
    GERMANY=Berlin
)

add_loader_test(
  loader_no_envp_test
  SRC
    main_without_envp.cpp
  DEPENDS
    libc.loader.linux.crt1
)

add_loader_test(
  loader_no_args_test
  SRC
    main_without_args.cpp
  DEPENDS
    libc.loader.linux.crt1
)
