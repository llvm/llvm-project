add_custom_target(libc-libgen-tests)

add_libc_test(
  basename_test
  SUITE
    libc-libgen-tests
  SRCS
    basename_test.cpp
  DEPENDS
    libc.src.libgen.basename
)

add_libc_test(
  dirname_test
  SUITE
    libc-libgen-tests
  SRCS
    dirname_test.cpp
  DEPENDS
    libc.src.libgen.dirname
)

if (NOT LLVM_USE_SANITIZER)
  add_libc_test(
    basename_death_test
    UNIT_TEST_ONLY
    SUITE
      libc-libgen-tests
    SRCS
      basename_death_test.cpp
    DEPENDS
      libc.src.libgen.basename
  )

  add_libc_test(
    dirname_death_test
    UNIT_TEST_ONLY
    SUITE
      libc-libgen-tests
    SRCS
      dirname_death_test.cpp
    DEPENDS
      libc.src.libgen.dirname
  )
endif()
