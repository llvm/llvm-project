
add_libc_unittest(
  file_test
  SUITE
    libc_support_unittests
  SRCS
    file_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.stdio
    libc.include.stdlib
    libc.src.__support.File.file
)

target_link_libraries(
  libc.test.src.__support.File.file_test PRIVATE LibcMemoryHelpers
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
