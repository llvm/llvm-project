add_entrypoint_object(
  ftw
  SRCS
    ftw.cpp
  HDRS
    ftw.h
  DEPENDS
    libc.include.ftw
    libc.src.__support.FPUtil.fenv_impl
    libc.src.errno.errno
    libc.src.fcntl.open
    libc.src.unistd.close

)
