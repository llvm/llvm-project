
add_entrypoint_library(
  llvmlibc
  DEPENDS
    # assert.h entrypoints
    libc.src.assert.__assert_fail

    # errno.h entrypoints
    libc.src.errno.__errno_location

    # signal.h entrypoints
    libc.src.signal.raise
    libc.src.signal.sigaction
    libc.src.signal.sigdelset
    libc.src.signal.sigaddset
    libc.src.signal.sigemptyset
    libc.src.signal.sigprocmask
    libc.src.signal.sigfillset
    libc.src.signal.signal

    # stdlib.h entrypoints
    libc.src.stdlib._Exit
    libc.src.stdlib.abort

    # string.h entrypoints
    libc.src.string.bzero
    libc.src.string.memcpy
    libc.src.string.memset
    libc.src.string.strcat
    libc.src.string.strcpy
    libc.src.string.strlen

    # sys/mman.h entrypoints
    libc.src.sys.mman.mmap
    libc.src.sys.mman.munmap

    # threads.h entrypoints
    libc.src.threads.mtx_init
    libc.src.threads.mtx_lock
    libc.src.threads.mtx_unlock
    libc.src.threads.thrd_create
    libc.src.threads.thrd_join

    # unistd.h entrypoints
    libc.src.unistd.write
)

add_entrypoint_library(
  llvmlibm
  DEPENDS
    # math.h entrypoints
    libc.src.math.cosf
    libc.src.math.fabs
    libc.src.math.fabsf
    libc.src.math.expf
    libc.src.math.exp2f
    libc.src.math.round
    libc.src.math.sincosf
    libc.src.math.sinf
)

add_redirector_library(
  llvmlibc_redirectors
  DEPENDS
    round_redirector
)
