add_custom_target(libc_wctype_unittests)

add_libc_test(
  iswalpha_test
  SUITE
    libc_wctype_unittests
  SRCS
    iswalpha_test.cpp
  DEPENDS
    libc.src.wctype.iswalpha
    libc.src.__support.wctype_utils
)

add_libc_test(
  iswdigit_test
  SUITE
    libc_wctype_unittests
  SRCS
    iswdigit_test.cpp
  DEPENDS
    libc.src.wctype.iswdigit
    libc.src.__support.wctype_utils
)
