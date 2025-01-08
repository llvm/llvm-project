function(add_fp_unittest name)
  cmake_parse_arguments(
    "MATH_UNITTEST"
    "NEED_MPFR;UNIT_TEST_ONLY;HERMETIC_TEST_ONLY" # Optional arguments
    "" # Single value arguments
    "LINK_LIBRARIES;DEPENDS" # Multi-value arguments
    ${ARGN}
  )

  if(MATH_UNITTEST_NEED_MPFR)
    if(NOT LIBC_TESTS_CAN_USE_MPFR)
      message(VERBOSE "Math test ${name} will be skipped as MPFR library is not available.")
      return()
    endif()
  endif()

  if(MATH_UNITTEST_HERMETIC_TEST_ONLY)
    set(test_type HERMETIC_TEST_ONLY)
  elseif(MATH_UNITTEST_UNIT_TEST_ONLY)
    set(test_type UNIT_TEST_ONLY)
  endif()
  if(MATH_UNITTEST_NEED_MPFR)
    if(MATH_UNITTEST_HERMETIC_TEST_ONLY)
      message(FATAL_ERROR "Hermetic math test cannot require MPFR.")
    endif()
    set(test_type UNIT_TEST_ONLY)
    list(APPEND MATH_UNITTEST_LINK_LIBRARIES libcMPFRWrapper -lmpfr -lgmp)
    if(NOT(LIBC_TARGET_OS_IS_DARWIN))
      # macOS does not have libatomic.
      list(APPEND MATH_UNITTEST_LINK_LIBRARIES -latomic)
    endif()
  endif()
  list(APPEND MATH_UNITTEST_LINK_LIBRARIES LibcFPTestHelpers)

  set(deps libc.hdr.math_macros)
  if(MATH_UNITTEST_DEPENDS)
    list(APPEND deps ${MATH_UNITTEST_DEPENDS})
  endif()

  add_libc_test(
    ${name}
    ${test_type}
    LINK_LIBRARIES "${MATH_UNITTEST_LINK_LIBRARIES}"
    "${MATH_UNITTEST_UNPARSED_ARGUMENTS}"
    DEPENDS "${deps}"
  )
endfunction(add_fp_unittest)

add_subdirectory(__support)
add_subdirectory(complex)
add_subdirectory(ctype)
add_subdirectory(errno)
add_subdirectory(fenv)
add_subdirectory(math)
add_subdirectory(search)
add_subdirectory(stdbit)
add_subdirectory(stdfix)
add_subdirectory(stdio)
add_subdirectory(stdlib)
add_subdirectory(string)
add_subdirectory(strings)
add_subdirectory(wchar)
add_subdirectory(time)

# Depends on utilities in stdlib
add_subdirectory(inttypes)

if(${LIBC_TARGET_OS} STREQUAL "linux")
  add_subdirectory(fcntl)
  add_subdirectory(sched)
  add_subdirectory(sys)
  add_subdirectory(termios)
  add_subdirectory(unistd)
endif()

if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

add_subdirectory(arpa)
add_subdirectory(assert)
add_subdirectory(compiler)
add_subdirectory(dirent)
add_subdirectory(locale)
add_subdirectory(setjmp)
add_subdirectory(signal)
add_subdirectory(spawn)

if(${LIBC_TARGET_OS} STREQUAL "linux")
  add_subdirectory(pthread)
endif()
