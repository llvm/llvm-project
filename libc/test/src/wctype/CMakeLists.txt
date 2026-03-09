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
  iswgraph_test
  SUITE
    libc_wctype_unittests
  SRCS
    iswgraph_test.cpp
  DEPENDS
    libc.src.wctype.iswgraph
)
