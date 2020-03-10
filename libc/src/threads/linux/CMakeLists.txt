add_gen_header(
  thread_start_args_h
  DEF_FILE thread_start_args.h.def
  GEN_HDR thread_start_args.h
  PARAMS
    thread_start_args=${LIBC_TARGET_MACHINE}/thread_start_args.h.in
  DATA_FILES
    ${LIBC_TARGET_MACHINE}/thread_start_args.h.in
)

add_header_library(
  threads_utils
  HDRS
    thread_utils.h
  DEPENDS
    thread_start_args_h
)

add_entrypoint_object(
  thrd_create
  SRCS
    thrd_create.cpp
  HDRS
    ../thrd_create.h
  DEPENDS
    errno_h
    linux_syscall_h
    mmap
    support_common_h
    sys_syscall_h
    threads_h
    threads_utils
    __errno_location
  COMPILE_OPTIONS
    -fno-omit-frame-pointer # This allows us to sniff out the thread args from
                            # the new thread's stack reliably.
)

add_entrypoint_object(
  thrd_join
  SRCS
    thrd_join.cpp
  HDRS
    ../thrd_join.h
  DEPENDS
    linux_syscall_h
    munmap
    support_common_h
    sys_syscall_h
    threads_h
    threads_utils
)
