add_gen_header(
  linux_syscall_h
  DEF_FILE syscall.h.def
  GEN_HDR syscall.h
  PARAMS
    inline_syscalls=${LIBC_TARGET_MACHINE}/syscall.h.inc
  DATA_FILES
    ${LIBC_TARGET_MACHINE}/syscall.h.inc
)

add_subdirectory(x86_64)
