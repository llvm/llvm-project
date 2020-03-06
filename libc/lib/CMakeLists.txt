
add_entrypoint_library(
  llvmlibc
  DEPENDS
    # errno.h entrypoints
    __errno_location

    # string.h entrypoints
    strcpy
    strcat

    # sys/mman.h entrypoints
    mmap
    munmap

    # signal.h entrypoints
    raise
    sigaddset
    sigemptyset
    sigprocmask

    # stdlib.h entrypoints
    _Exit
    abort

    # threads.h entrypoints
    thrd_create
    thrd_join
)

add_entrypoint_library(
  llvmlibm
  DEPENDS
    # math.h entrypoints
    round
)

add_redirector_library(
  llvmlibc_redirectors
  DEPENDS
    round_redirector
)
