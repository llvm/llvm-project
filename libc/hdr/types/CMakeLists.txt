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
  struct_timespec
  HDRS
    struct_timespec.h
  FULL_BUILD_DEPENDS
    libc.include.llvm-libc-types.struct_timespec
)
