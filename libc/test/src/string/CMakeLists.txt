add_libc_testsuite(libc_string_unittests)

add_subdirectory(memory_utils)

add_libc_unittest(
  bcopy_test
  SUITE
    libc_string_unittests
  SRCS
    bcopy_test.cpp
  DEPENDS
    libc.src.string.bcopy
  LINK_LIBRARIES
    LibcMemoryHelpers
)

add_libc_unittest(
  memccpy_test
  SUITE
    libc_string_unittests
  SRCS
    memccpy_test.cpp
  DEPENDS
    libc.src.string.memccpy
)

add_libc_unittest(
  mempcpy_test
  SUITE
    libc_string_unittests
  SRCS
    mempcpy_test.cpp
  DEPENDS
    libc.src.string.mempcpy
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
  memrchr_test
  SUITE
    libc_string_unittests
  SRCS
    memrchr_test.cpp
  DEPENDS
    libc.src.string.memrchr
)

add_libc_unittest(
  stpcpy_test
  SUITE
    libc_string_unittests
  SRCS
    stpcpy_test.cpp
  DEPENDS
    libc.src.string.stpcpy
)

add_libc_unittest(
  stpncpy_test
  SUITE
    libc_string_unittests
  SRCS
    stpncpy_test.cpp
  DEPENDS
    libc.src.string.stpncpy
)

add_libc_unittest(
  strcat_test
  SUITE
    libc_string_unittests
  SRCS
    strcat_test.cpp
  DEPENDS
    libc.src.string.strcat
)

add_header_library(
  strchr_test_support
  HDRS
    StrchrTest.h
)

add_libc_unittest(
  strchr_test
  SUITE
    libc_string_unittests
  SRCS
    strchr_test.cpp
  DEPENDS
    libc.src.string.strchr
    .strchr_test_support
)

add_libc_unittest(
  strchrnul_test
  SUITE
    libc_string_unittests
  SRCS
    strchrnul_test.cpp
  DEPENDS
    libc.src.string.strchrnul
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
  strcasecmp_test
  SUITE
    libc_string_unittests
  SRCS
    strcasecmp_test.cpp
  DEPENDS
    libc.src.string.strcasecmp
)

add_libc_unittest(
  strcasestr_test
  SUITE
    libc_string_unittests
  SRCS
    strcasestr_test.cpp
  DEPENDS
    libc.src.string.strcasestr
)

add_libc_unittest(
  strcoll_test
  SUITE
    libc_string_unittests
  SRCS
    strcoll_test.cpp
  DEPENDS
    libc.src.string.strcoll
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
  strcspn_test
  SUITE
    libc_string_unittests
  SRCS
    strcspn_test.cpp
  DEPENDS
    libc.src.string.strcspn
)

add_libc_unittest(
  strdup_test
  SUITE
    libc_string_unittests
  SRCS
    strdup_test.cpp
  DEPENDS
    libc.include.stdlib
    libc.src.string.strdup
    libc.src.errno.errno
)

add_libc_unittest(
  strerror_test
  SUITE
    libc_string_unittests
  SRCS
    strerror_test.cpp
  DEPENDS
    libc.src.string.strerror
)


add_libc_unittest(
  strerror_r_test
  SUITE
    libc_string_unittests
  SRCS
    strerror_r_test.cpp
  DEPENDS
    libc.src.string.strerror_r
)

add_libc_unittest(
  strlcat_test
  SUITE
    libc_string_unittests
  SRCS
    strlcat_test.cpp
  DEPENDS
    libc.src.string.strlcat
)

add_libc_unittest(
  strlcpy_test
  SUITE
    libc_string_unittests
  SRCS
    strlcpy_test.cpp
  DEPENDS
    libc.src.string.strlcpy
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
  strncat_test
  SUITE
    libc_string_unittests
  SRCS
    strncat_test.cpp
  DEPENDS
    libc.src.string.strncat
)

add_libc_unittest(
  strncmp_test
  SUITE
    libc_string_unittests
  SRCS
    strncmp_test.cpp
  DEPENDS
    libc.src.string.strncmp
)

add_libc_unittest(
  strncasecmp_test
  SUITE
    libc_string_unittests
  SRCS
    strncasecmp_test.cpp
  DEPENDS
    libc.src.string.strncasecmp
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
  strndup_test
  SUITE
    libc_string_unittests
  SRCS
    strndup_test.cpp
  DEPENDS
    libc.include.stdlib
    libc.src.string.strndup
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
  strpbrk_test
  SUITE
    libc_string_unittests
  SRCS
    strpbrk_test.cpp
  DEPENDS
    libc.src.string.strpbrk
)

add_libc_unittest(
  strrchr_test
  SUITE
    libc_string_unittests
  SRCS
    strrchr_test.cpp
  DEPENDS
    libc.src.string.strrchr
    .strchr_test_support
)

add_libc_unittest(
  strsep_test
  SUITE
    libc_string_unittests
  SRCS
    strsep_test.cpp
  DEPENDS
    libc.src.string.strsep
)

add_libc_unittest(
  strsignal_test
  SUITE
    libc_string_unittests
  SRCS
    strsignal_test.cpp
  DEPENDS
    libc.src.string.strsignal
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
  strstr_test
  SUITE
    libc_string_unittests
  SRCS
    strstr_test.cpp
  DEPENDS
    libc.src.string.strstr
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

add_libc_unittest(
  strxfrm_test
  SUITE
    libc_string_unittests
  SRCS
    strxfrm_test.cpp
  DEPENDS
    libc.src.string.strxfrm
)

# Tests all implementations that can run on the target CPU.
function(add_libc_multi_impl_test name)
  get_property(fq_implementations GLOBAL PROPERTY ${name}_implementations)
  foreach(fq_config_name IN LISTS fq_implementations)
    get_target_property(required_cpu_features ${fq_config_name} REQUIRE_CPU_FEATURES)
    cpu_supports(can_run "${required_cpu_features}")
    if(can_run)
      add_libc_unittest(
        ${fq_config_name}_test
        SUITE
          libc_string_unittests
        COMPILE_OPTIONS
          ${LIBC_COMPILE_OPTIONS_NATIVE}
        LINK_LIBRARIES
          LibcMemoryHelpers
        ${ARGN}
        DEPENDS
          ${fq_config_name}
          libc.src.__support.macros.sanitizer
      )
      get_fq_target_name(${fq_config_name}_test fq_target_name)
    else()
      message(STATUS "Skipping test for '${fq_config_name}' insufficient host cpu features '${required_cpu_features}'")
    endif()
  endforeach()
endfunction()

add_libc_multi_impl_test(bcmp SRCS bcmp_test.cpp)
add_libc_multi_impl_test(bzero SRCS bzero_test.cpp)
add_libc_multi_impl_test(memcmp SRCS memcmp_test.cpp)
add_libc_multi_impl_test(memcpy SRCS memcpy_test.cpp)
add_libc_multi_impl_test(memmove SRCS memmove_test.cpp)
add_libc_multi_impl_test(memset SRCS memset_test.cpp)
