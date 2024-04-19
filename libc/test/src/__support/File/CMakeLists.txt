if(NOT (TARGET libc.src.__support.threads.mutex) OR LIBC_TARGET_OS_IS_GPU)
  # Not all platforms have a mutex implementation. If mutex is unvailable,
  # we just skip everything about files. The GPU does not currently support
  # files as well.
  return()
endif()

add_libc_test(
  file_test
  SUITE
    libc-support-tests
  SRCS
    file_test.cpp
  LINK_LIBRARIES
    LibcMemoryHelpers
  DEPENDS
    libc.include.errno
    libc.include.stdio
    libc.include.stdlib
    libc.src.__support.CPP.new
    libc.src.__support.File.file
)

add_libc_test(
  platform_file_test
  SUITE
    libc-support-tests
  SRCS
    platform_file_test.cpp
  DEPENDS
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
    libc.include.stdio
)

set(platform_stream_targets "platform_stdout;platform_stdin;platform_stderr")
foreach(target IN LISTS platform_file_targets)
  if(TARGET libc.src.__support.File.${target})
    add_libc_test(
      ${target}_test
      HERMETIC_TEST_ONLY
      SUITE
        libc-support-tests
      SRCS
        ${target}_test.cpp
      DEPENDS
        libc.src.__support.File.file
        libc.src.__support.File.${target}
        libc.include.stdio
    )
  endif()
endforeach()

add_subdirectory(testdata)
