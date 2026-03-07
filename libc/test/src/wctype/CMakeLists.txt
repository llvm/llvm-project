add_custom_target(libc_wctype_unittests)

add_libc_test(
  iswalpha_test
  SUITE
    libc_wctype_unittests
  SRCS
    iswalpha_test.cpp
  DEPENDS
    libc.src.wctype.iswalpha
)

add_libc_test(
  iswlower_test
  SUITE
    libc_wctype_unittests
  SRCS
    iswlower_test.cpp
  DEPENDS
    libc.src.wctype.iswlower
)
