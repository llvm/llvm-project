add_header_library(
  posix_semaphore
  HDRS
    posix_semaphore.h
  DEPENDS
    libc.src.__support.CPP.atomic
    libc.src.__support.threads.futex_utils
)
