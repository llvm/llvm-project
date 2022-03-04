
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
