add_libc_testsuite(libc-startup-tests PARENT libc-integration-tests)

add_integration_test(
  startup_no_args_test
  SUITE libc-startup-tests
  SRCS
    main_without_args.cpp
)
