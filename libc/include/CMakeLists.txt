
add_header(
  llvm_libc_common_h
  HDR
    __llvm-libc-common.h
)

add_header(
  libc_posix_types
  HDR
    __posix-types.h
)

add_header(
  stdc_types
  HDR
    __llvm-libc-stdc-types.h
)

add_gen_header(
  ctype
  DEF_FILE ctype.h.def
  GEN_HDR ctype.h
  DEPENDS
    .llvm_libc_common_h
)

add_gen_header(
  fenv
  DEF_FILE fenv.h.def
  GEN_HDR fenv.h
  DEPENDS
    .llvm_libc_common_h
)

add_gen_header(
  inttypes
  DEF_FILE inttypes.h.def
  GEN_HDR inttypes.h
  DEPENDS
    .llvm_libc_common_h
)

add_gen_header(
  math
  DEF_FILE math.h.def
  GEN_HDR math.h
  DEPENDS
    .llvm_libc_common_h
    .stdc_types
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
)

add_gen_header(
  time
  DEF_FILE time.h.def
  GEN_HDR time.h
  DEPENDS
    .llvm_libc_common_h
)

add_gen_header(
  threads
  DEF_FILE threads.h.def
  GEN_HDR threads.h
  PARAMS
    platform_threads=../config/${LIBC_TARGET_OS}/threads.h.in
  DEPENDS
    .llvm_libc_common_h
  DATA_FILES
    ../config/${LIBC_TARGET_OS}/threads.h.in
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
)

add_gen_header(
  stdio
  DEF_FILE stdio.h.def
  GEN_HDR stdio.h
  DEPENDS
    .llvm_libc_common_h
)

add_gen_header(
  stdlib
  DEF_FILE stdlib.h.def
  GEN_HDR stdlib.h
  DEPENDS
    .llvm_libc_common_h
)

add_gen_header(
  unistd
  DEF_FILE unistd.h.def
  GEN_HDR unistd.h
  DEPENDS
    .libc_posix_types
    .llvm_libc_common_h
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
    .libc_posix_types
    .llvm_libc_common_h
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
