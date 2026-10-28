add_libc_testsuite(libc-support-stringutil-tests PARENT libc-support-tests)

add_libc_test(
  error_to_string_test
  SUITE
    libc-support-stringutil-tests
  SRCS
    error_to_string_test.cpp
  DEPENDS
    libc.src.__support.CPP.span
    libc.src.__support.CPP.string_view
    libc.src.__support.StringUtil.error_to_string
)

if(LIBC_TARGET_OS_IS_LINUX AND TARGET libc.src.__support.StringUtil.signal_to_string)
  add_libc_test(
    signal_to_string_test
    SUITE
      libc-support-stringutil-tests
    SRCS
      signal_to_string_test.cpp
    DEPENDS
      libc.hdr.signal_macros
      libc.src.__support.CPP.span
      libc.src.__support.CPP.string_view
      libc.src.__support.StringUtil.signal_to_string
      libc.src.__support.integer_to_string
  )
endif()
