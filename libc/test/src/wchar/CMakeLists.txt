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

add_libc_test(
  wcsncmp_test
  SUITE
    libc_wchar_unittests
  SRCS
    wcsncmp_test.cpp
  DEPENDS
    libc.src.wchar.wcsncmp
)

add_libc_test(
  wcscmp_test
  SUITE
    libc_wchar_unittests
  SRCS
    wcscmp_test.cpp
  DEPENDS
    libc.src.wchar.wcscmp
)

add_libc_test(
  wcspbrk_test
  SUITE
    libc_wchar_unittests
  SRCS
    wcspbrk_test.cpp
  DEPENDS
    libc.src.wchar.wcspbrk
)

add_libc_test(
  wcsrchr_test
  SUITE
    libc_wchar_unittests
  SRCS
    wcsrchr_test.cpp
  DEPENDS
    libc.src.wchar.wcsrchr
)

add_libc_test(
  wcsspn_test
  SUITE
    libc_wchar_unittests
  SRCS
    wcsspn_test.cpp
  DEPENDS
    libc.src.wchar.wcsspn
)

add_libc_test(
  wmemchr_test
  SUITE
    libc_wchar_unittests
  SRCS
    wmemchr_test.cpp
  DEPENDS
    libc.src.wchar.wmemchr
)

add_libc_test(
  wmemcmp_test
  SUITE
    libc_wchar_unittests
  SRCS
    wmemcmp_test.cpp
  DEPENDS
    libc.src.wchar.wmemcmp
)

add_libc_test(
  wmempcpy_test
  SUITE
    libc_wchar_unittests
  SRCS
    wmempcpy_test.cpp
  DEPENDS
    libc.src.wchar.wmempcpy
)

add_libc_test(
  wmemcpy_test
  SUITE
    libc_wchar_unittests
  SRCS
    wmemcpy_test.cpp
  DEPENDS
    libc.src.wchar.wmemcpy
)

add_libc_test(
  wmemmove_test
  SUITE
    libc_wchar_unittests
  SRCS
    wmemmove_test.cpp
  DEPENDS
    libc.src.wchar.wmemmove
)

add_libc_test(
  wcsncpy_test
  SUITE
    libc_wchar_unittests
  SRCS
    wcsncpy_test.cpp
  DEPENDS
    libc.src.wchar.wcsncpy
)

add_libc_test(
  wcscat_test
  SUITE
    libc_wchar_unittests
  SRCS
    wcscat_test.cpp
  DEPENDS
    libc.src.wchar.wcscat
)
  
add_libc_test(
  wcsstr_test
  SUITE
    libc_wchar_unittests
  SRCS
    wcsstr_test.cpp
  DEPENDS
    libc.src.wchar.wcsstr
)

add_libc_test(
  wcsncat_test
  SUITE
    libc_wchar_unittests
  SRCS
    wcsncat_test.cpp
  DEPENDS
    libc.src.wchar.wcsncat
)

add_libc_test(
  wcscpy_test
  SUITE
    libc_wchar_unittests
  SRCS
    wcscpy_test.cpp
  DEPENDS
    libc.src.wchar.wcscpy
)
