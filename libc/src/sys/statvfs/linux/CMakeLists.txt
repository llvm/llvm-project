add_header_library(
  statfs_utils
  HDRS
    statfs_utils.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.OSUtil.osutil
    libc.src.__support.common
    libc.src.__support.CPP.optional
    libc.include.sys_syscall
)

add_entrypoint_object(
  statvfs
  SRCS
    statvfs.cpp
  HDRS
    ../statvfs.h
  DEPENDS
    libc.src.__support.libc_assert
    libc.include.llvm-libc-types.struct_statvfs
    .statfs_utils
)

add_entrypoint_object(
  fstatvfs
  SRCS
    fstatvfs.cpp
  HDRS
    ../fstatvfs.h
  DEPENDS
    libc.src.__support.libc_assert
    libc.include.llvm-libc-types.struct_statvfs
    .statfs_utils
)

