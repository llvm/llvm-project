add_libc_testsuite(libc_string_unittests)

add_subdirectory(memory_utils)

add_libc_unittest(
  strcat_test
  SUITE
    libc_string_unittests
  SRCS
    strcat_test.cpp
  DEPENDS
    libc.src.string.strcat
)

add_libc_unittest(
  strcpy_test
  SUITE
    libc_string_unittests
  SRCS
    strcpy_test.cpp
  DEPENDS
    libc.src.string.strcpy
)

add_libc_unittest(
  strlen_test
  SUITE
    libc_string_unittests
  SRCS
    strlen_test.cpp
  DEPENDS
    libc.src.string.strlen
)

add_libc_unittest(
  strcmp_test
  SUITE
    libc_string_unittests
  SRCS
    strcmp_test.cpp
  DEPENDS
    libc.src.string.strcmp
)

# Tests all implementations that can run on the host.
function(add_libc_multi_impl_test name)
  get_property(fq_implementations GLOBAL PROPERTY ${name}_implementations)
  foreach(fq_config_name IN LISTS fq_implementations)
    get_target_property(required_cpu_features ${fq_config_name} REQUIRE_CPU_FEATURES)
    host_supports(can_run "${required_cpu_features}")
    if(can_run)
      add_libc_unittest(
        ${fq_config_name}_test
        SUITE
          libc_string_unittests
        DEPENDS
          ${fq_config_name}
        ${ARGN}
      )
    else()
      message(STATUS "Skipping test for '${fq_config_name}' insufficient host cpu features '${required_cpu_features}'")
    endif()
  endforeach()
endfunction()

add_libc_multi_impl_test(memcpy SRCS memcpy_test.cpp)
add_libc_multi_impl_test(memset SRCS memset_test.cpp)
add_libc_multi_impl_test(bzero SRCS bzero_test.cpp)

