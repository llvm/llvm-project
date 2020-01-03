add_custom_target(libc_string_unittests)
add_dependencies(check_libc libc_string_unittests)

add_libc_unittest(
  strcat_test
  SUITE
    libc_string_unittests
  SRCS
    strcat_test.cpp
  DEPENDS
    strcat
    strcpy
)

add_libc_unittest(
  strcpy_test
  SUITE
    libc_string_unittests
  SRCS
    strcpy_test.cpp
  DEPENDS
    strcpy
)
