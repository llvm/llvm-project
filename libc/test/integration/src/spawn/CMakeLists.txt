add_custom_target(spawn-integration-tests)
add_dependencies(libc-integration-tests spawn-integration-tests)

add_executable(
  libc_posix_spawn_test_binary
  EXCLUDE_FROM_ALL
  posix_spawn_test_binary.cpp
  test_binary_properties.h
)
set_target_properties(
  libc_posix_spawn_test_binary
  PROPERTIES
    OUTPUT_NAME libc_posix_spawn_test_binary
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_header_library(
  test_binary_properties
  HDRS
    test_binary_properties.h
)

add_integration_test(
  posix_spawn_test
  SUITE
    spawn-integration-tests
  SRCS
    posix_spawn_test.cpp
  LOADER
    libc.loader.linux.crt1
  DEPENDS
    libc_posix_spawn_test_binary
    libc.test.integration.src.spawn.test_binary_properties
    libc.include.fcntl
    libc.include.signal
    libc.include.spawn
    libc.include.sys_wait
    libc.src.signal.raise
    libc.src.spawn.posix_spawn
    libc.src.spawn.posix_spawn_file_actions_addopen
    libc.src.spawn.posix_spawn_file_actions_destroy
    libc.src.spawn.posix_spawn_file_actions_init
    libc.src.sys.wait.waitpid
)

add_subdirectory(testdata)
