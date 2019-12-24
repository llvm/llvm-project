
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
