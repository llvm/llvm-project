add_entrypoint_object(
  uname
  SRCS
    uname.cpp
  HDRS
    ../uname.h
  DEPENDS
    libc.include.sys_utsname
    libc.include.sys_syscall
    libc.src.__support.OSUtil.osutil
    libc.src.errno.errno
)

