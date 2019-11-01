
add_entrypoint_library(
  llvmlibc
  DEPENDS
    # string.h entrypoints
    ## C standard library entrypoints
    strcpy
    strcat
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
