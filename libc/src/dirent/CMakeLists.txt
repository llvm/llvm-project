if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()


add_entrypoint_object(
  opendir
  SRCS
    opendir.cpp
  HDRS
    opendir.h
  DEPENDS
    libc.include.dirent
    libc.src.__support.File.dir
    libc.src.__support.File.platform_dir
    libc.src.errno.errno
)

add_entrypoint_object(
  dirfd
  SRCS
    dirfd.cpp
  HDRS
    dirfd.h
  DEPENDS
    libc.include.dirent
    libc.src.__support.File.dir
    libc.src.__support.File.platform_dir
)

add_entrypoint_object(
  closedir
  SRCS
    closedir.cpp
  HDRS
    closedir.h
  DEPENDS
    libc.include.dirent
    libc.src.__support.File.dir
    libc.src.__support.File.platform_dir
    libc.src.errno.errno
)

add_entrypoint_object(
  readdir
  SRCS
    readdir.cpp
  HDRS
    readdir.h
  DEPENDS
    libc.include.dirent
    libc.src.__support.File.dir
    libc.src.__support.File.platform_dir
    libc.src.errno.errno
)

add_entrypoint_object(
  fdopendir
  SRCS
    fdopendir.cpp
  HDRS
    fdopendir.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.macros.config
    libc.hdr.types.DIR
    libc.src.__support.File.dir
    libc.src.errno.errno
)

add_entrypoint_object(
  scandir
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.scandir
)
