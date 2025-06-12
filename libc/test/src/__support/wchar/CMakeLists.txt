add_custom_target(libc-support-wchar-tests)

add_libc_test(
  utf8_to_32_test 
  SUITE
    libc-support-tests
  SRCS
    utf8_to_32_test.cpp 
  DEPENDS
    libc.src.__support.wchar.character_converter
)