add_proxy_header_library(
  sigset_t
  HDRS
    sigset_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.sigset_t
)

add_proxy_header_library(
  struct_epoll_event
  HDRS
    struct_epoll_event.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.struct_epoll_event
)

add_proxy_header_library(
  struct_flock
  HDRS
    struct_flock.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.struct_flock
)

add_proxy_header_library(
  struct_flock64
  HDRS
    struct_flock64.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.struct_flock64
)

add_proxy_header_library(
  struct_f_owner_ex
  HDRS
    struct_f_owner_ex.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.struct_f_owner_ex
)

add_proxy_header_library(
  struct_timespec
  HDRS
    struct_timespec.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.struct_timespec
)

add_proxy_header_library(
  fenv_t
  HDRS
    fenv_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.fenv_t
    libc.include.fenv
)

add_proxy_header_library(
  fexcept_t
  HDRS
    fexcept_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.fexcept_t
    libc.include.fenv
)

add_proxy_header_library(
  time_t
  HDRS
    time_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.time_t
    libc.include.time
)

add_proxy_header_library(
  clockid_t
  HDRS
    clockid_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.clockid_t
    libc.include.sys_types
)

add_proxy_header_library(
  clock_t
  HDRS
    clock_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.clock_t
    libc.include.time
)

add_proxy_header_library(
  suseconds_t
  HDRS
    suseconds_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.suseconds_t
    libc.include.sys_time
)

add_proxy_header_library(
  struct_timeval
  HDRS
    struct_timeval.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.struct_timeval
    libc.include.sys_time
)

add_proxy_header_library(
  pid_t
  HDRS
    pid_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.pid_t
    libc.include.sys_types
)

add_proxy_header_library(
  atexithandler_t
  HDRS
    atexithandler_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.atexithandler_t
    libc.include.stdlib
)

add_proxy_header_library(
  struct_sigaction
  HDRS
    struct_sigaction.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.struct_sigaction
    libc.include.signal
)

add_proxy_header_library(
  FILE
  HDRS
    FILE.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.FILE
    libc.include.stdio
)

add_proxy_header_library(
  off_t
  HDRS
    off_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.off_t
    libc.include.stdio
)

add_proxy_header_library(
  cookie_io_functions_t
  HDRS
    cookie_io_functions_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.cookie_io_functions_t
    libc.include.stdio
)

add_proxy_header_library(
  locale_t
  HDRS
    locale_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.locale_t
    libc.include.locale
)

add_proxy_header_library(
  sighandler_t
  HDRS
    sighandler_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.__sighandler_t
    libc.include.signal
)

add_proxy_header_library(
  stack_t
  HDRS
    stack_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.stack_t
    libc.include.signal
)

add_proxy_header_library(
  jmp_buf
  HDRS
    jmp_buf.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.jmp_buf
    libc.include.setjmp
)


add_proxy_header_library(
  struct_msghdr
  HDRS
    struct_msghdr.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.struct_msghdr
    libc.include.sys_socket
)

add_proxy_header_library(
  struct_sockaddr
  HDRS
    struct_sockaddr.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.struct_sockaddr
    libc.include.sys_socket
)

add_proxy_header_library(
  socklen_t
  HDRS
    socklen_t.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.socklen_t
    libc.include.sys_socket
)
