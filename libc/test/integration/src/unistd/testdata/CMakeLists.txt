add_executable(
  libc_execvp_test_normal_exit
  EXCLUDE_FROM_ALL
  execvp_test_normal_exit.cpp
)
set_target_properties(
  libc_execvp_test_normal_exit
  PROPERTIES
    OUTPUT_NAME libc_execvp_test_normal_exit
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
