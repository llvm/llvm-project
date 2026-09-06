add_libc_testsuite(libc-execinfo-tests)

add_libc_test(
  backtrace_test
  SUITE
    libc-execinfo-tests
  SRCS
    backtrace_test.cpp
  COMPILE_OPTIONS
    # The unwinder reads the tables the compiler emits for each frame, and
    # the tests are built without them by default.
    -funwind-tables
    # The frames the test builds have to stay separate for there to be three
    # of them to walk back through.
    -fno-optimize-sibling-calls
  DEPENDS
    libc.hdr.fcntl_macros
    libc.hdr.sys_stat_macros
    libc.src.execinfo.backtrace
    libc.src.execinfo.backtrace_symbols_fd
    libc.src.fcntl.open
    libc.src.sys.stat.fstat
    libc.src.unistd.close
    libc.src.unistd.unlink
)

add_libc_test(
  backtrace_symbols_test
  SUITE
    libc-execinfo-tests
  SRCS
    backtrace_symbols_test.cpp
  DEPENDS
    libc.hdr.types.size_t
    libc.src.execinfo.backtrace_symbols
    libc.src.string.strlen
    libc.src.string.strstr
)
