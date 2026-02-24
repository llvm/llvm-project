add_custom_target(libc-string-tests)

add_subdirectory(memory_utils)

add_header_library(
  strchr_test_support
  HDRS
    StrchrTest.h
)

add_libc_test(
  memccpy_test
  SUITE
    libc-string-tests
  SRCS
    memccpy_test.cpp
  DEPENDS
    libc.src.string.memccpy
)

add_libc_test(
  mempcpy_test
  SUITE
    libc-string-tests
  SRCS
    mempcpy_test.cpp
  DEPENDS
    libc.src.string.mempcpy
)

add_libc_test(
  memmem_test
  SUITE
    libc-string-tests
  SRCS
    memmem_test.cpp
  DEPENDS
    libc.src.string.memmem
)

add_libc_test(
  memchr_test
  SUITE
    libc-string-tests
  SRCS
    memchr_test.cpp
  DEPENDS
    libc.src.string.memchr
)

add_libc_test(
  memrchr_test
  SUITE
    libc-string-tests
  SRCS
    memrchr_test.cpp
  DEPENDS
    libc.src.string.memrchr
)

add_libc_test(
  stpcpy_test
  SUITE
    libc-string-tests
  SRCS
    stpcpy_test.cpp
  DEPENDS
    libc.src.string.stpcpy
)

add_libc_test(
  stpncpy_test
  SUITE
    libc-string-tests
  SRCS
    stpncpy_test.cpp
  DEPENDS
    libc.src.string.stpncpy
)

add_libc_test(
  strcat_test
  SUITE
    libc-string-tests
  SRCS
    strcat_test.cpp
  DEPENDS
    libc.src.string.strcat
)

add_libc_test(
  strchr_test
  SUITE
    libc-string-tests
  SRCS
    strchr_test.cpp
  DEPENDS
    libc.src.string.strchr
    .strchr_test_support
)

add_libc_test(
  strchrnul_test
  SUITE
    libc-string-tests
  SRCS
    strchrnul_test.cpp
  DEPENDS
    libc.src.string.strchrnul
)

add_libc_test(
  strcmp_test
  SUITE
    libc-string-tests
  SRCS
    strcmp_test.cpp
  DEPENDS
    libc.src.string.strcmp
)

add_libc_test(
  strcasestr_test
  SUITE
    libc-string-tests
  SRCS
    strcasestr_test.cpp
  DEPENDS
    libc.src.string.strcasestr
)

add_libc_test(
  strcoll_test
  SUITE
    libc-string-tests
  SRCS
    strcoll_test.cpp
  DEPENDS
    libc.src.string.strcoll
)

add_libc_test(
  strcpy_test
  SUITE
    libc-string-tests
  SRCS
    strcpy_test.cpp
  DEPENDS
    libc.src.string.strcpy
)

add_libc_test(
  strcspn_test
  SUITE
    libc-string-tests
  SRCS
    strcspn_test.cpp
  DEPENDS
    libc.src.string.strcspn
)

add_libc_test(
  strdup_test
  SUITE
    libc-string-tests
  SRCS
    strdup_test.cpp
  DEPENDS
    libc.src.string.strdup
    libc.src.errno.errno
    libc.test.UnitTest.ErrnoCheckingTest
)

# FIXME: This is failing on the bot for some reason, disable for now.
if(NOT LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  add_libc_test(
    strerror_test
    SUITE
      libc-string-tests
    SRCS
      strerror_test.cpp
    DEPENDS
      libc.src.string.strerror
  )
endif()

add_libc_test(
  strerror_r_test
  SUITE
    libc-string-tests
  SRCS
    strerror_r_test.cpp
  DEPENDS
    libc.src.string.strerror_r
)

add_libc_test(
  strlcat_test
  SUITE
    libc-string-tests
  SRCS
    strlcat_test.cpp
  DEPENDS
    libc.src.string.strlcat
)

add_libc_test(
  strlcpy_test
  SUITE
    libc-string-tests
  SRCS
    strlcpy_test.cpp
  DEPENDS
    libc.src.string.strlcpy
)

add_libc_test(
  strlen_test
  SUITE
    libc-string-tests
  SRCS
    strlen_test.cpp
  DEPENDS
    libc.src.string.strlen
)

add_libc_test(
  strncat_test
  SUITE
    libc-string-tests
  SRCS
    strncat_test.cpp
  DEPENDS
    libc.src.string.strncat
)

add_libc_test(
  strncmp_test
  SUITE
    libc-string-tests
  SRCS
    strncmp_test.cpp
  DEPENDS
    libc.src.string.strncmp
)

add_libc_test(
  strncpy_test
  SUITE
    libc-string-tests
  SRCS
    strncpy_test.cpp
  DEPENDS
    libc.src.string.strncpy
)

add_libc_test(
  strndup_test
  SUITE
    libc-string-tests
  SRCS
    strndup_test.cpp
  DEPENDS
    libc.src.string.strndup
)

add_libc_test(
  strnlen_test
  SUITE
    libc-string-tests
  SRCS
    strnlen_test.cpp
  DEPENDS
    libc.src.string.strnlen
)

add_libc_test(
  strpbrk_test
  SUITE
    libc-string-tests
  SRCS
    strpbrk_test.cpp
  DEPENDS
    libc.src.string.strpbrk
)

add_libc_test(
  strrchr_test
  SUITE
    libc-string-tests
  SRCS
    strrchr_test.cpp
  DEPENDS
    libc.src.string.strrchr
    .strchr_test_support
)

add_libc_test(
  strsep_test
  SUITE
    libc-string-tests
  SRCS
    strsep_test.cpp
  DEPENDS
    libc.src.string.strsep
)

add_libc_test(
  strsignal_test
  SUITE
    libc-string-tests
  SRCS
    strsignal_test.cpp
  DEPENDS
    libc.src.string.strsignal
)

add_libc_test(
  strspn_test
  SUITE
    libc-string-tests
  SRCS
    strspn_test.cpp
  DEPENDS
    libc.src.string.strspn
)

add_libc_test(
  strstr_test
  SUITE
    libc-string-tests
  SRCS
    strstr_test.cpp
  DEPENDS
    libc.src.string.strstr
)

add_libc_test(
  strtok_test
  SUITE
    libc-string-tests
  SRCS
    strtok_test.cpp
  DEPENDS
    libc.src.string.strtok
)

add_libc_test(
  strtok_r_test
  SUITE
    libc-string-tests
  SRCS
    strtok_r_test.cpp
  DEPENDS
    libc.src.string.strtok_r
)

add_libc_test(
  strxfrm_test
  SUITE
    libc-string-tests
  SRCS
    strxfrm_test.cpp
  DEPENDS
    libc.src.string.strxfrm
)

add_libc_test(
  memset_explicit_test
  SUITE
    libc-string-tests
  SRCS
    memset_explicit_test.cpp
  DEPENDS
    libc.src.string.memset_explicit
)

add_libc_multi_impl_test(memcmp libc-string-tests SRCS memcmp_test.cpp)
add_libc_multi_impl_test(memcpy libc-string-tests SRCS memcpy_test.cpp)
add_libc_multi_impl_test(memmove libc-string-tests SRCS memmove_test.cpp)
add_libc_multi_impl_test(memset libc-string-tests SRCS memset_test.cpp)
