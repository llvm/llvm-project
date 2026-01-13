add_entrypoint_object(
  clock_gettime
  SRCS
    clock_gettime.cpp
  HDRS
    # The public header is part of the parent directory's library.
  DEPENDS
    libc.src.__support.time.clock_gettime
    libc.src.errno.errno
)
