add_libc_testsuite(libc_string_unittests)

add_subdirectory(memory_utils)

add_libc_unittest(
  strcat_test
  SUITE
    libc_string_unittests
  SRCS
    strcat_test.cpp
  DEPENDS
    strcat
    strcpy
    strlen
)

add_libc_unittest(
  strcpy_test
  SUITE
    libc_string_unittests
  SRCS
    strcpy_test.cpp
  DEPENDS
    strcpy
    strlen
)

add_libc_unittest(
  strlen_test
  SUITE
    libc_string_unittests
  SRCS
    strlen_test.cpp
  DEPENDS
    strlen
)

# Tests all implementations of memcpy that can run on the host.
get_property(memcpy_implementations GLOBAL PROPERTY memcpy_implementations)
foreach(memcpy_config_name IN LISTS memcpy_implementations)
  get_target_property(require_cpu_features ${memcpy_config_name} REQUIRE_CPU_FEATURES)
  host_supports(can_run "${require_cpu_features}")
  if(can_run)
    add_libc_unittest(
      ${memcpy_config_name}_test
      SUITE
        libc_string_unittests
      SRCS
        memcpy_test.cpp
      DEPENDS
        ${memcpy_config_name}
    )
  else()
    message(STATUS "Skipping test for '${memcpy_config_name}' insufficient host cpu features")
  endif()
endforeach()

