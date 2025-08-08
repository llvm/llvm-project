add_custom_target(stdlib-linux-integration-tests)
add_dependencies(libc-integration-tests stdlib-linux-integration-tests)

add_integration_test(
  vsdo_rng_test
  SUITE
    stdlib-linux-integration-tests
  SRCS
    vsdo_rng_test.cpp
  DEPENDS
    libc.src.pthread.pthread_create
    libc.src.pthread.pthread_join
    libc.src.stdlib.linux.vsdo_rng
)
