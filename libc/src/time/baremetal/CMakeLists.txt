add_entrypoint_object(
  timespec_get
  SRCS
    timespec_get.cpp
  HDRS
    ../timespec_get.h
  DEPENDS
    libc.hdr.time_macros
    libc.hdr.types.struct_timespec
)

add_entrypoint_object(
  localtime
  SRCS
    localtime.cpp
  HDRS
    ../localtime.h
  DEPENDS
    libc.hdr.time_macros
    libc.hdr.types.struct_timespec
)

