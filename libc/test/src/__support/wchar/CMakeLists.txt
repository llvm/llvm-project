add_custom_target(libc-support-wchar-tests)

add_libc_test(
  utf32_to_8_test
  SUITE
    libc-support-tests
  SRCS
    utf32_to_8_test.cpp
  DEPENDS
    libc.src.__support.wchar.character_converter
)
