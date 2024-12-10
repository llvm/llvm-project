add_entrypoint_object(
  getrandom
  SRCS
    getrandom.cpp
  HDRS
    ../getrandom.h
  DEPENDS
    # Maybe we should include the following 
    # but we don't really care the flags on Windows
    # libc.hdr.sys_random_macros
    libc.hdr.types.ssize_t
    libc.src.errno.errno
    libc.src.__support.CPP.limits
)
