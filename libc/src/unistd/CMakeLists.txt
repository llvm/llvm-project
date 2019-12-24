add_gen_header(
  syscall_impl_h
  DEF_FILE syscall.h.def
  GEN_HDR syscall.h
  PARAMS
    inline_syscalls=../../config/${LIBC_TARGET_OS}/${LIBC_TARGET_MACHINE}/syscall.h.inc
  DATA_FILES
    ../../config/${LIBC_TARGET_OS}/${LIBC_TARGET_MACHINE}/syscall.h.inc
)
