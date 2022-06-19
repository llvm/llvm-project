if(NOT (TARGET libc.src.__support.threads.mutex))
  # Not all platforms have a mutex implementation. If mutex is unvailable,
  # we just skip everything about files.
  return()
endif()

add_libc_unittest(
  file_test
  SUITE
    libc_support_unittests
  SRCS
    file_test.cpp
  LINK_LIBRARIES
    LibcMemoryHelpers
  DEPENDS
    libc.include.errno
    libc.include.stdio
    libc.include.stdlib
    libc.src.__support.File.file
)

if (TARGET libc.src.__support.File.platform_file)
  add_libc_unittest(
    platform_file_test
    SUITE
      libc_support_unittests
    SRCS
      platform_file_test.cpp
    DEPENDS
      libc.src.__support.File.file
      libc.src.__support.File.platform_file
      libc.include.stdio
  )
endif()

add_subdirectory(testdata)
