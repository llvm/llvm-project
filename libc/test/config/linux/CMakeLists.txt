add_libc_testsuite(libc_linux_tests)

if(EXISTS ${LIBC_TARGET_MACHINE})
  add_subdirectory(${LIBC_TARGET_MACHINE})
endif()
