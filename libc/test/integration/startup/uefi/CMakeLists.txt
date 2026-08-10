add_custom_target(libc-startup-tests)
add_dependencies(libc-integration-tests libc-startup-tests)

add_integration_test(
  startup_no_args_test
  SUITE libc-startup-tests
  SRCS
    main_without_args.cpp
)
