add_libc_testsuite(libc_sys_resource_unittests)

add_subdirectory(testdata)

add_libc_unittest(
  getrlimit_setrlimit_test
  SUITE
    libc_sys_resource_unittests
  SRCS
    getrlimit_setrlimit_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.fcntl
    libc.include.sys_resource
    libc.src.fcntl.open
    libc.src.sys.resource.getrlimit
    libc.src.sys.resource.setrlimit
    libc.src.unistd.close
    libc.src.unistd.unlink
)
