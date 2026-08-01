add_custom_target(libc_pwd_unittests)

add_libc_unittest(
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
