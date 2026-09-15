add_custom_target(libc_grp_unittests)

if(NOT TARGET libc.src.grp.grp_utils)
  return()
endif()

add_libc_test(
  grp_utils_test
  SUITE
    libc_grp_unittests
  SRCS
    grp_utils_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.types.gid_t
    libc.hdr.types.size_t
    libc.hdr.types.struct_group
    libc.src.__support.CPP.span
    libc.src.__support.pwd.flat_file_db
    libc.src.grp.grp_utils
)
