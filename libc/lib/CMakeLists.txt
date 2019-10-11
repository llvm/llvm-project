
add_entrypoint_library(
  llvmlibc
  DEPENDS
    ### C standard library entrypoints
    # string.h entrypoints
    strcpy
    strcat
)
