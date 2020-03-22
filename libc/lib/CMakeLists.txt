
add_entrypoint_library(
  llvmlibc
  DEPENDS
    # assert.h entrypoints
    __assert_fail

    # errno.h entrypoints
    __errno_location

    # string.h entrypoints
    strcpy
    strcat
    memcpy

    # sys/mman.h entrypoints
    mmap
    munmap

    # signal.h entrypoints
    raise
    sigaction
    sigaddset
    sigemptyset
    sigprocmask
    signal

    # stdlib.h entrypoints
    _Exit
    abort

    # threads.h entrypoints
    mtx_init
    mtx_lock
    mtx_unlock
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
