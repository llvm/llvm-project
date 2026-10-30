add_libc_testsuite(libc-sys-ptrace-integration-tests PARENT libc-integration-tests)

add_integration_test(
  ptrace_test
  SUITE
    libc-sys-ptrace-integration-tests
  SRCS
    ptrace_test.cpp
  DEPENDS
    libc.include.sys_ptrace
    libc.hdr.signal_macros
    libc.src.sys.ptrace.ptrace
    libc.src.unistd.fork
    libc.src.signal.raise
    libc.src.signal.kill
    libc.src.sys.wait.waitpid
    libc.src.stdlib.exit
    libc.src.errno.errno
)
