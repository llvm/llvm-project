add_custom_target(libc-stdlib-tests)

add_libc_test(
  atof_test
  SUITE
    libc-stdlib-tests
  SRCS
    atof_test.cpp
  DEPENDS
    libc.src.errno.errno
    libc.src.stdlib.atof
    libc.test.errno_setter_matcher
)

add_header_library(
  atoi_test_support
  HDRS
    AtoiTest.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.CPP.type_traits
)

add_libc_test(
  atoi_test
  SUITE
    libc-stdlib-tests
  SRCS
    atoi_test.cpp
  DEPENDS
    .atoi_test_support
    libc.src.stdlib.atoi
)

add_libc_test(
  atol_test
  SUITE
    libc-stdlib-tests
  SRCS
    atol_test.cpp
  DEPENDS
    .atoi_test_support
    libc.src.stdlib.atol
)

add_libc_test(
  atoll_test
  SUITE
    libc-stdlib-tests
  SRCS
    atoll_test.cpp
  DEPENDS
    .atoi_test_support
    libc.src.stdlib.atoll
)

# This fails on NVPTX where the output value is one-off of the expected value.
if(NOT LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX)
  add_fp_unittest(
    strtod_test
    SUITE
      libc-stdlib-tests
    SRCS
      strtod_test.cpp
    DEPENDS
      libc.src.errno.errno
      libc.src.stdlib.strtod
      libc.src.__support.FPUtil.fenv_impl
  )
endif()

add_fp_unittest(
  strtof_test
  SUITE
    libc-stdlib-tests
  SRCS
    strtof_test.cpp
  DEPENDS
    libc.src.errno.errno
    libc.src.stdlib.strtof
    libc.src.__support.FPUtil.fenv_impl
)

add_header_library(
  strtol_test_support
  HDRS
    StrtolTest.h
  DEPENDS
    libc.src.__support.CPP.limits
    libc.src.__support.CPP.type_traits
    libc.src.errno.errno
)

add_libc_test(
  strtoint32_test
  SUITE
    libc-stdlib-tests
  SRCS
    strtoint32_test.cpp
  DEPENDS
    libc.src.__support.str_to_integer
    libc.src.errno.errno
    .strtol_test_support
)

add_libc_test(
  strtoint64_test
  SUITE
    libc-stdlib-tests
  SRCS
    strtoint64_test.cpp
  DEPENDS
    libc.src.__support.str_to_integer
    libc.src.errno.errno
    .strtol_test_support
)

add_libc_test(
  strtol_test
  SUITE
    libc-stdlib-tests
  SRCS
    strtol_test.cpp
  DEPENDS
    libc.src.stdlib.strtol
    .strtol_test_support
)

# This fails on NVPTX where the output value is one-off of the expected value.
if(NOT LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX)
  add_libc_test(
    strtold_test
    SUITE
      libc-stdlib-tests
    SRCS
      strtold_test.cpp
    DEPENDS
      libc.src.errno.errno
      libc.src.__support.uint128
      libc.src.stdlib.strtold
  )
endif()

add_libc_test(
  strtoll_test
  SUITE
    libc-stdlib-tests
  SRCS
    strtoll_test.cpp
  DEPENDS
    libc.src.stdlib.strtoll
    .strtol_test_support
)

add_libc_test(
  strtoul_test
  SUITE
    libc-stdlib-tests
  SRCS
    strtoul_test.cpp
  DEPENDS
    libc.src.stdlib.strtoul
    .strtol_test_support
)

add_libc_test(
  strtoull_test
  SUITE
    libc-stdlib-tests
  SRCS
    strtoull_test.cpp
  DEPENDS
    libc.src.stdlib.strtoull
    .strtol_test_support
)

add_libc_test(
  abs_test
  SUITE
    libc-stdlib-tests
  SRCS
    abs_test.cpp
  DEPENDS
    libc.src.stdlib.abs
)

add_libc_test(
  labs_test
  SUITE
    libc-stdlib-tests
  SRCS
    labs_test.cpp
  DEPENDS
    libc.src.stdlib.labs
)

add_libc_test(
  llabs_test
  SUITE
    libc-stdlib-tests
  SRCS
    llabs_test.cpp
  DEPENDS
    libc.src.stdlib.llabs
)

add_libc_test(
  div_test
  SUITE
    libc-stdlib-tests
  SRCS
    div_test.cpp
  HDRS
    DivTest.h
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib.div
)

add_libc_test(
  ldiv_test
  SUITE
    libc-stdlib-tests
  SRCS
    ldiv_test.cpp
  HDRS
    DivTest.h
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib.ldiv
)

add_libc_test(
  lldiv_test
  SUITE
    libc-stdlib-tests
  SRCS
    lldiv_test.cpp
  HDRS
    DivTest.h
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib.lldiv
)

add_libc_test(
  bsearch_test
  SUITE
    libc-stdlib-tests
  SRCS
    bsearch_test.cpp
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib.bsearch
)

add_libc_test(
  qsort_test
  SUITE
    libc-stdlib-tests
  SRCS
    qsort_test.cpp
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib.qsort
)


add_libc_test(
  qsort_r_test
  SUITE
    libc-stdlib-tests
  SRCS
    qsort_r_test.cpp
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib.qsort_r
)

add_libc_test(
  rand_test
  SUITE
    libc-stdlib-tests
  SRCS
    rand_test.cpp
  DEPENDS
    libc.include.stdlib
    libc.src.stdlib.rand
    libc.src.stdlib.srand
)

if(LLVM_LIBC_FULL_BUILD)

  add_libc_test(
    _Exit_test
    # The EXPECT_EXITS test is only availible for unit tests.
    UNIT_TEST_ONLY
    SUITE
      libc-stdlib-tests
    SRCS
      _Exit_test.cpp
    DEPENDS
      libc.include.stdlib
      libc.src.stdlib._Exit
      libc.src.stdlib.exit
  )

  add_libc_test(
    atexit_test
    # The EXPECT_EXITS test is only availible for unit tests.
    UNIT_TEST_ONLY
    SUITE
      libc-stdlib-tests
    SRCS
      atexit_test.cpp
    DEPENDS
      libc.include.stdlib
      libc.src.stdlib._Exit
      libc.src.stdlib.exit
      libc.src.stdlib.atexit
      libc.src.__support.CPP.array
      libc.src.__support.CPP.utility
  )

  add_libc_test(
    abort_test
    # The EXPECT_DEATH test is only availible for unit tests.
    UNIT_TEST_ONLY
    SUITE
      libc-stdlib-tests
    SRCS
      abort_test.cpp
    DEPENDS
      libc.include.stdlib
      libc.include.signal
      libc.src.stdlib.abort
      libc.src.stdlib._Exit
      libc.src.signal.raise
  )

  # Only the GPU has an in-tree 'malloc' implementation.
  if(LIBC_TARGET_ARCHITECTURE_IS_GPU)
    add_libc_test(
      malloc_test
      HERMETIC_TEST_ONLY
      SUITE
        libc-stdlib-tests
      SRCS
        malloc_test.cpp
      DEPENDS
        libc.include.stdlib
        libc.src.stdlib.malloc
        libc.src.stdlib.free
    )
  endif()
endif()
