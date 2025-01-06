add_custom_target(libc_wchar_unittests)

add_libc_test(
  btowc_test
  SUITE
    libc_wchar_unittests
  SRCS
    btowc_test.cpp
  DEPENDS
    libc.include.stdio
    libc.src.wchar.btowc
)

add_libc_test(
  wctob_test
  SUITE
    libc_wchar_unittests
  SRCS
    wctob_test.cpp
  DEPENDS
    libc.src.wchar.wctob
)
