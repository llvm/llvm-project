add_entrypoint_object(
  freeaddrinfo
  SRCS
    freeaddrinfo.cpp
  HDRS
    freeaddrinfo.h
  DEPENDS
    libc.hdr.types.struct_addrinfo
    libc.src.__support.common
    libc.src.__support.macros.config
)

add_entrypoint_object(
  gai_strerror
  SRCS
    gai_strerror.cpp
  HDRS
    gai_strerror.h
  DEPENDS
    libc.hdr.netdb_macros
    libc.src.__support.common
    libc.src.__support.macros.config
)

add_entrypoint_object(
  getaddrinfo
  SRCS
    getaddrinfo.cpp
  HDRS
    getaddrinfo.h
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.netdb_macros
    libc.hdr.types.struct_addrinfo
    libc.src.__support.common
    libc.src.__support.macros.config
    libc.src.errno.errno
)

if(NOT TARGET libc.src.__support.File.file OR
   NOT TARGET libc.src.__support.File.platform_file)
  return()
endif()

add_entrypoint_object(
  getprotobyname_r
  SRCS
    getprotobyname_r.cpp
  HDRS
    getprotobyname_r.h
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.stdint_proxy
    libc.hdr.types.size_t
    libc.hdr.types.struct_protoent
    libc.src.__support.CPP.optional
    libc.src.__support.CPP.scope
    libc.src.__support.CPP.span
    libc.src.__support.CPP.string_view
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
    libc.src.__support.ctype_utils
    libc.src.__support.macros.null_check
    libc.src.__support.str_to_integer
    libc.src.string.memory_utils.inline_memcpy
)
