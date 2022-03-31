add_custom_target(stdlib-integration-tests)
add_dependencies(libc-integration-tests stdlib-integration-tests)

add_integration_test(
  getenv_test
  SUITE
    stdlib-integration-tests
  SRCS
    getenv_test.cpp
  LOADER
    libc.loader.linux.crt1
  DEPENDS
    libc.src.stdlib.getenv
  ENV
    FRANCE=Paris
    GERMANY=Berlin
)

