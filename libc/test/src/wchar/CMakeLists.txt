add_custom_target(libc_wchar_unittests)

add_libc_test(
  wcslen_test
  SUITE
    libc_wchar_unittests
  SRCS
    wcslen_test.cpp
  DEPENDS
    libc.hdr.types.size_t
    libc.hdr.types.wchar_t
    libc.src.wchar.wcslen
)

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

add_libc_test(
  wmemset_test 
  SUITE
    libc_wchar_unittests
  SRCS
    wmemset_test.cpp
  DEPENDS
    libc.hdr.types.size_t
    libc.hdr.types.wchar_t
    libc.src.wchar.wmemset
)

add_libc_test(
  wcschr_test
  SUITE
    libc_wchar_unittests
  SRCS
    wcschr_test.cpp
  DEPENDS
    libc.src.wchar.wcschr
)
