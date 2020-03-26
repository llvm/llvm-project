add_loader_test(
  loader_args_test
  SRC
    args_test.cpp
  DEPENDS
    __assert_fail
    _Exit
    abort
    crt1
    raise
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
    crt1
)

add_loader_test(
  loader_no_args_test
  SRC
    main_without_args.cpp
  DEPENDS
    crt1
)
