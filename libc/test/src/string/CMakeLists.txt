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

add_libc_unittest(
  memchr_test
  SUITE
    libc_string_unittests
  SRCS
    memchr_test.cpp
  DEPENDS
    libc.src.string.memchr
)

add_libc_unittest(
  memcmp_test
  SUITE
    libc_string_unittests
  SRCS
    memcmp_test.cpp
  DEPENDS
    libc.src.string.memcmp
)

add_libc_unittest(
  strchr_test
  SUITE
    libc_string_unittests
  SRCS
    strchr_test.cpp
  DEPENDS
    libc.src.string.strchr
)

add_libc_unittest(
  strstr_test
  SUITE
    libc_string_unittests
  SRCS
    strstr_test.cpp
  DEPENDS
    libc.src.string.strstr
)

add_libc_unittest(
  strncpy_test
  SUITE
    libc_string_unittests
  SRCS
    strncpy_test.cpp
  DEPENDS
    libc.src.string.strncpy
)

add_libc_unittest(
  strnlen_test
  SUITE
    libc_string_unittests
  SRCS
    strnlen_test.cpp
  DEPENDS
    libc.src.string.strnlen
)

add_libc_unittest(
  memrchr_test
  SUITE
    libc_string_unittests
  SRCS
    memrchr_test.cpp
  DEPENDS
    libc.src.string.memrchr
)

add_libc_unittest(
  strrchr_test
  SUITE
    libc_string_unittests
  SRCS
    strrchr_test.cpp
  DEPENDS
    libc.src.string.strrchr
)

add_libc_unittest(
  strcspn_test
  SUITE
    libc_string_unittests
  SRCS
    strcspn_test.cpp
  DEPENDS
    libc.src.string.strcspn
)

add_libc_unittest(
  strspn_test
  SUITE
    libc_string_unittests
  SRCS
    strspn_test.cpp
  DEPENDS
    libc.src.string.strspn
)

add_libc_unittest(
  strpbrk_test
  SUITE
    libc_string_unittests
  SRCS
    strpbrk_test.cpp
  DEPENDS
    libc.src.string.strpbrk
)

add_libc_unittest(
  strtok_test
  SUITE
    libc_string_unittests
  SRCS
    strtok_test.cpp
  DEPENDS
    libc.src.string.strtok
)

add_libc_unittest(
  strtok_r_test
  SUITE
    libc_string_unittests
  SRCS
    strtok_r_test.cpp
  DEPENDS
    libc.src.string.strtok_r
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
