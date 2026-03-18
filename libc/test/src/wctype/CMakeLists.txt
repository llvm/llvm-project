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

add_libc_test(
  iswcntrl_test
  SUITE
    libc_wctype_unittests
  SRCS
    iswcntrl_test.cpp
  DEPENDS
    libc.src.wctype.iswcntrl
)

add_libc_test(
  iswgraph_test
  SUITE
    libc_wctype_unittests
  SRCS
    iswgraph_test.cpp
  DEPENDS
    libc.src.wctype.iswgraph
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

add_libc_test(
  iswspace_test
  SUITE
    libc_wctype_unittests
  SRCS
    iswspace_test.cpp
  DEPENDS
    libc.src.wctype.iswspace
)

add_libc_test(
  iswblank_test
  SUITE
    libc_wctype_unittests
  SRCS
    iswblank_test.cpp
  DEPENDS
    libc.src.wctype.iswblank
)

add_libc_test(
  iswxdigit_test
  SUITE
    libc_wctype_unittests
  SRCS
    iswxdigit_test.cpp
  DEPENDS
    libc.src.wctype.iswxdigit
)

add_libc_test(
  iswalnum_test
  SUITE
    libc_wctype_unittests
  SRCS
    iswalnum_test.cpp
  DEPENDS
    libc.src.wctype.iswalnum
)

add_libc_test(
  iswupper_test
  SUITE
    libc_wctype_unittests
  SRCS
    iswupper_test.cpp
  DEPENDS
    libc.src.wctype.iswupper
)

add_libc_test(
  iswpunct_test
  SUITE
    libc_wctype_unittests
  SRCS
    iswpunct_test.cpp
  DEPENDS
    libc.src.wctype.iswpunct
)
