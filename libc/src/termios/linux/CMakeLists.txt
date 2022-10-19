# There is no difference between input and output speeds on Linux.
# However, since POSIX requires separate functions for setting and getting
# of the input and output speeds, we use different entrypoints wiht the
# same getter/setter logic.
add_entrypoint_object(
  cfgetispeed
  SRCS
    cfgetispeed.cpp
  HDRS
    ../cfgetispeed.h
  DEPENDS
    libc.include.termios
)

add_entrypoint_object(
  cfsetispeed
  SRCS
    cfsetispeed.cpp
  HDRS
    ../cfsetispeed.h
  DEPENDS
    libc.include.termios
    libc.src.errno.errno
)

add_entrypoint_object(
  cfgetospeed
  SRCS
    cfgetospeed.cpp
  HDRS
    ../cfgetospeed.h
  DEPENDS
    libc.include.termios
)

add_entrypoint_object(
  cfsetospeed
  SRCS
    cfsetospeed.cpp
  HDRS
    ../cfsetospeed.h
  DEPENDS
    libc.include.termios
    libc.src.errno.errno
)

add_entrypoint_object(
  tcgetsid
  SRCS
    tcgetsid.cpp
  HDRS
    ../tcgetsid.h
  DEPENDS
    libc.include.sys_syscall
    libc.include.termios
    libc.src.__support.OSUtil.osutil
    libc.src.errno.errno
)

add_entrypoint_object(
  tcdrain
  SRCS
    tcdrain.cpp
  HDRS
    ../tcdrain.h
  DEPENDS
    libc.include.sys_syscall
    libc.include.termios
    libc.src.__support.OSUtil.osutil
    libc.src.errno.errno
)

add_entrypoint_object(
  tcflush
  SRCS
    tcflush.cpp
  HDRS
    ../tcflush.h
  DEPENDS
    libc.include.sys_syscall
    libc.include.termios
    libc.src.__support.OSUtil.osutil
    libc.src.errno.errno
)

add_entrypoint_object(
  tcflow
  SRCS
    tcflow.cpp
  HDRS
    ../tcflow.h
  DEPENDS
    libc.include.sys_syscall
    libc.include.termios
    libc.src.__support.OSUtil.osutil
    libc.src.errno.errno
)

add_entrypoint_object(
  tcsendbreak
  SRCS
    tcsendbreak.cpp
  HDRS
    ../tcsendbreak.h
  DEPENDS
    libc.include.sys_syscall
    libc.include.termios
    libc.src.__support.OSUtil.osutil
    libc.src.errno.errno
)

add_header_library(
  kernel_termios
  HDRS
    kernel_termios.h
)

add_entrypoint_object(
  tcgetattr
  SRCS
    tcgetattr.cpp
  HDRS
    ../tcgetattr.h
  DEPENDS
    .kernel_termios
    libc.include.sys_syscall
    libc.include.termios
    libc.src.__support.OSUtil.osutil
    libc.src.errno.errno
)

add_entrypoint_object(
  tcsetattr
  SRCS
    tcsetattr.cpp
  HDRS
    ../tcsetattr.h
  DEPENDS
    .kernel_termios
    libc.include.sys_syscall
    libc.include.termios
    libc.src.__support.OSUtil.osutil
    libc.src.errno.errno
)
