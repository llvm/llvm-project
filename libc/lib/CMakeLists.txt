include("${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/${LIBC_TARGET_MACHINE}/entrypoints.txt")
include("${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/${LIBC_TARGET_MACHINE}/headers.txt")

add_entrypoint_library(
  llvmlibc
  DEPENDS
  ${LIBC_ENTRYPOINTS}
)

add_entrypoint_library(
  llvmlibm
  DEPENDS
  ${LIBM_ENTRYPOINTS}
)

add_redirector_library(
  llvmlibc_redirectors
  DEPENDS
    round_redirector
)
