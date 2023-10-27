add_custom_target(libc_sys_sendfile_unittests)

add_subdirectory(testdata)

add_libc_unittest(
  sendfile_test
  SUITE
    libc_sys_sendfile_unittests
  SRCS
    sendfile_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.fcntl
    libc.include.sys_stat
    libc.src.fcntl.open
    libc.src.sys.sendfile.sendfile
    libc.src.unistd.close
    libc.src.unistd.read
    libc.src.unistd.unlink
    libc.src.unistd.write
)
