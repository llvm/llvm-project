add_entrypoint_library(
  llvmlibc
  DEPENDS
  ${TARGET_LIBC_ENTRYPOINTS}
)

add_entrypoint_library(
  llvmlibm
  DEPENDS
  ${TARGET_LIBM_ENTRYPOINTS}
)
