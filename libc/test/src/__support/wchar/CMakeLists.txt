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

add_libc_test(
  utf32_to_8_test
  SUITE
    libc-support-tests
  SRCS
    utf32_to_8_test.cpp
  DEPENDS
    libc.src.__support.wchar.character_converter
)


add_libc_test(
  string_converter_test
  SUITE
    libc-support-tests
  SRCS
    string_converter_test.cpp
  DEPENDS
    libc.src.__support.wchar.string_converter
    libc.src.__support.wchar.mbstate
    libc.src.__support.error_or
    libc.hdr.errno_macros
    libc.hdr.types.char32_t
)
