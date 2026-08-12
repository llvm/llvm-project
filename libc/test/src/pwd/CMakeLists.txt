add_custom_target(libc_pwd_unittests)

if(NOT TARGET libc.src.pwd.pwd_utils)
  return()
endif()

add_libc_test(
  pwd_utils_test
  SUITE
    libc_pwd_unittests
  SRCS
    pwd_utils_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.types.struct_passwd
    libc.src.pwd.pwd_utils
)

add_libc_test(
  getpwent_test
  SUITE
    libc_pwd_unittests
  SRCS
    getpwent_test.cpp
  DEPENDS
    libc.hdr.types.struct_passwd
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
    libc.src.errno.errno
    libc.src.pwd.endpwent
    libc.src.pwd.getpwent
    libc.src.pwd.pwd_utils
    libc.src.pwd.setpwent
    libc.src.stdio.remove
    libc.src.string.string_utils
)
