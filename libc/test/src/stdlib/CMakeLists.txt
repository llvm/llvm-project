add_libc_testsuite(libc_stdlib_unittests)

add_libc_unittest(
  atof_test
  SUITE
    libc_stdlib_unittests
  SRCS
    atof_test.cpp
  DEPENDS
    libc.src.stdlib.atof
)

add_libc_unittest(
  atoi_test
  SUITE
    libc_stdlib_unittests
  SRCS
    atoi_test.cpp
  DEPENDS
    libc.src.stdlib.atoi
)

add_libc_unittest(
  atol_test
  SUITE
    libc_stdlib_unittests
  SRCS
    atol_test.cpp
  DEPENDS
    libc.src.stdlib.atol
)

add_libc_unittest(
  atoll_test
  SUITE
    libc_stdlib_unittests
  SRCS
    atoll_test.cpp
  DEPENDS
    libc.src.stdlib.atoll
)

add_libc_unittest(
  strtod_test
  SUITE
    libc_stdlib_unittests
  SRCS
    strtod_test.cpp
  DEPENDS
    libc.src.stdlib.strtod
)

add_libc_unittest(
  strtof_test
  SUITE
    libc_stdlib_unittests
  SRCS
    strtof_test.cpp
  DEPENDS
    libc.src.stdlib.strtof
)

add_libc_unittest(
  strtol_test
  SUITE
    libc_stdlib_unittests
  SRCS
    strtol_test.cpp
  DEPENDS
    libc.src.stdlib.strtol
)

add_libc_unittest(
  strtold_test
  SUITE
    libc_stdlib_unittests
  SRCS
    strtold_test.cpp
  DEPENDS
    libc.src.stdlib.strtold
)

add_libc_unittest(
  strtoll_test
  SUITE
    libc_stdlib_unittests
  SRCS
    strtoll_test.cpp
  DEPENDS
    libc.src.stdlib.strtoll
)

add_libc_unittest(
  strtoul_test
  SUITE
    libc_stdlib_unittests
  SRCS
    strtoul_test.cpp
  DEPENDS
    libc.src.stdlib.strtoul
)

add_libc_unittest(
  strtoull_test
  SUITE
    libc_stdlib_unittests
  SRCS
    strtoull_test.cpp
  DEPENDS
    libc.src.stdlib.strtoull
)

if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

add_libc_unittest(
  _Exit_test
  SUITE
    libc_stdlib_unittests
  SRCS
    _Exit_test.cpp
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib._Exit
)

# add_libc_unittest(
#   abort_test
#   SUITE
#     libc_stdlib_unittests
#   SRCS
#     abort_test.cpp
#   DEPENDS
#     libc.include.stdlib
#     libc.include.signal
#     libc.src.stdlib.abort
#     libc.src.stdlib._Exit
#     libc.src.signal.raise
# )

add_libc_unittest(
  abs_test
  SUITE
    libc_stdlib_unittests
  SRCS
    abs_test.cpp
  DEPENDS
    libc.src.stdlib.abs
)

add_libc_unittest(
  labs_test
  SUITE
    libc_stdlib_unittests
  SRCS
    labs_test.cpp
  DEPENDS
    libc.src.stdlib.labs
)

add_libc_unittest(
  llabs_test
  SUITE
    libc_stdlib_unittests
  SRCS
    llabs_test.cpp
  DEPENDS
    libc.src.stdlib.llabs
)

add_libc_unittest(
  div_test
  SUITE
    libc_stdlib_unittests
  SRCS
    div_test.cpp
  HDRS
    DivTest.h
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib.div
)

add_libc_unittest(
  ldiv_test
  SUITE
    libc_stdlib_unittests
  SRCS
    ldiv_test.cpp
  HDRS
    DivTest.h
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib.ldiv
)

add_libc_unittest(
  lldiv_test
  SUITE
    libc_stdlib_unittests
  SRCS
    lldiv_test.cpp
  HDRS
    DivTest.h
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib.lldiv
)

add_libc_unittest(
  bsearch_test
  SUITE
    libc_stdlib_unittests
  SRCS
    bsearch_test.cpp
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib.bsearch
)

add_libc_unittest(
  qsort_test
  SUITE
    libc_stdlib_unittests
  SRCS
    qsort_test.cpp
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib.qsort
)
