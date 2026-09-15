if(NOT TARGET libc.src.__support.File.file OR
   NOT TARGET libc.src.__support.File.platform_file)
  return()
endif()

add_object_library(
  grp_utils
  HDRS
    grp_utils.h
  SRCS
    grp_utils.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.stdint_proxy
    libc.hdr.types.gid_t
    libc.hdr.types.size_t
    libc.hdr.types.struct_group
    libc.src.__support.CPP.span
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
    libc.src.__support.ctype_utils
    libc.src.__support.error_or
    libc.src.__support.macros.attributes
    libc.src.__support.macros.config
    libc.src.__support.pwd.dynamic_buffer
    libc.src.__support.pwd.field_tokenizer
    libc.src.__support.pwd.flat_file_db
    libc.src.__support.str_to_integer
)
