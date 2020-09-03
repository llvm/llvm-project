add_libc_testsuite(libc_linux_tests)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_MACHINE})
  add_subdirectory(${LIBC_TARGET_MACHINE})
endif()
