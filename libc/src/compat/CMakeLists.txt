# Names the platform ABI settled on that are not what the standards call the
# functions behind them. Code already built, and code that states its own
# declarations, asks for these.

add_entrypoint_object(
  syscall
  SRCS
    syscall.cpp
  HDRS
    syscall.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.OSUtil.osutil
    libc.src.__support.common
    libc.src.__support.libc_errno
    libc.src.__support.macros.config
    libc.src.__support.macros.properties.architectures
)
