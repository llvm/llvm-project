add_subdirectory(llvm-libc-macros)
add_subdirectory(llvm-libc-types)

add_header(
  llvm_libc_common_h
  HDR
    __llvm-libc-common.h
)

add_gen_header(
  ctype
  DEF_FILE ctype.h.def
  GEN_HDR ctype.h
  DEPENDS
    .llvm_libc_common_h
)

add_gen_header(
  fcntl
  DEF_FILE fcntl.h.def
  GEN_HDR fcntl.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.fcntl_macros
    .llvm-libc-types.mode_t
)

add_gen_header(
  fenv
  DEF_FILE fenv.h.def
  GEN_HDR fenv.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.fenv_t
    .llvm-libc-types.fexcept_t
)

add_gen_header(
  inttypes
  DEF_FILE inttypes.h.def
  GEN_HDR inttypes.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.imaxdiv_t
)

add_gen_header(
  math
  DEF_FILE math.h.def
  GEN_HDR math.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.double_t
    .llvm-libc-types.float_t
)

add_gen_header(
  assert_h
  DEF_FILE assert.h.def
  GEN_HDR assert.h
  DEPENDS
    .llvm_libc_common_h
)

add_gen_header(
  string
  DEF_FILE string.h.def
  GEN_HDR string.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.size_t
)

add_gen_header(
  time
  DEF_FILE time.h.def
  GEN_HDR time.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.time_t
    .llvm-libc-types.struct_tm
)

add_gen_header(
  threads
  DEF_FILE threads.h.def
  GEN_HDR threads.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.__call_once_func_t
    .llvm-libc-types.cnd_t
    .llvm-libc-types.mtx_t
    .llvm-libc-types.once_flag
    .llvm-libc-types.thrd_start_t
    .llvm-libc-types.thrd_t
)

add_gen_header(
  errno
  DEF_FILE errno.h.def
  PARAMS
    platform_errno=../config/${LIBC_TARGET_OS}/errno.h.in
  GEN_HDR errno.h
  DATA_FILES
    ../config/${LIBC_TARGET_OS}/errno.h.in
)

add_gen_header(
  signal
  DEF_FILE signal.h.def
  PARAMS
    platform_signal=../config/${LIBC_TARGET_OS}/signal.h.in
  GEN_HDR signal.h
  DATA_FILES
    ../config/${LIBC_TARGET_OS}/signal.h.in
  DEPENDS
    .llvm-libc-types.struct_sigaction
    .llvm-libc-types.__sighandler_t
)

add_gen_header(
  stdio
  DEF_FILE stdio.h.def
  GEN_HDR stdio.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-macros.stdio_macros
    .llvm-libc-types.FILE
    .llvm-libc-types.size_t
)

add_gen_header(
  stdlib
  DEF_FILE stdlib.h.def
  GEN_HDR stdlib.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.__bsearchcompare_t
    .llvm-libc-types.__qsortcompare_t
    .llvm-libc-types.div_t
    .llvm-libc-types.ldiv_t
    .llvm-libc-types.lldiv_t
    .llvm-libc-types.size_t
)

add_gen_header(
  unistd
  DEF_FILE unistd.h.def
  GEN_HDR unistd.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.size_t
    .llvm-libc-types.ssize_t
)

# TODO: Not all platforms will have a include/sys directory. Add the sys
# directory and the targets for sys/*.h files conditional to the OS requiring
# them.
file(MAKE_DIRECTORY "sys")

add_gen_header(
  sys_mman
  DEF_FILE sys/mman.h.def
  GEN_HDR sys/mman.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.off_t
    .llvm-libc-types.ssize_t
)

add_gen_header(
  sys_stat
  DEF_FILE sys/stat.h.def
  GEN_HDR sys/stat.h
  DEPENDS
    .llvm_libc_common_h
    .llvm-libc-types.mode_t
)

add_gen_header(
  sys_syscall
  DEF_FILE sys/syscall.h.def
  GEN_HDR sys/syscall.h
  PARAMS
    syscall_numbers=../config/${LIBC_TARGET_OS}/syscall_numbers.h.inc
  DATA_FILES
    ../config/${LIBC_TARGET_OS}/syscall_numbers.h.inc
)
