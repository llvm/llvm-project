if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_header_library(
  time_constants
  HDRS
    time_constants.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.hdr.types.time_t
    libc.include.time
    libc.src.__support.CPP.array
    libc.src.__support.CPP.string_view
)

add_object_library(
  time_utils
  SRCS
    time_utils.cpp
  HDRS
    time_utils.h
  DEPENDS
    libc.include.time
    libc.src.__support.CPP.limits
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.optional
    libc.src.errno.errno
    .time_constants
    libc.hdr.types.time_t
    libc.hdr.types.size_t
    libc.hdr.types.struct_tm
    libc.hdr.stdint_proxy
)

add_entrypoint_object(
  asctime
  SRCS
    asctime.cpp
  HDRS
    asctime.h
  DEPENDS
    .time_utils
    .time_constants
    libc.include.time
    libc.hdr.types.struct_tm
)

add_entrypoint_object(
  asctime_r
  SRCS
    asctime_r.cpp
  HDRS
    asctime_r.h
  DEPENDS
    .time_utils
    .time_constants
    libc.include.time
    libc.hdr.types.struct_tm
)

add_entrypoint_object(
  ctime
  SRCS
    ctime.cpp
  HDRS
    ctime.h
  DEPENDS
    .time_utils
    .time_constants
    libc.hdr.types.time_t
    libc.include.time
)

add_entrypoint_object(
  ctime_r
  SRCS
    ctime_r.cpp
  HDRS
    ctime_r.h
  DEPENDS
    .time_utils
    .time_constants
    libc.hdr.types.time_t
    libc.include.time
)

add_entrypoint_object(
  localtime
  SRCS
    localtime.cpp
  HDRS
    localtime.h
  DEPENDS
    .time_utils
    libc.hdr.types.time_t
    libc.hdr.types.struct_tm
)

add_entrypoint_object(
  localtime_r
  SRCS
    localtime_r.cpp
  HDRS
    localtime_r.h
  DEPENDS
    .time_utils
    libc.hdr.types.time_t
    libc.hdr.types.struct_tm
)

add_entrypoint_object(
  difftime
  SRCS
    difftime.cpp
  HDRS
    difftime.h
  DEPENDS
    libc.include.time
    libc.hdr.types.time_t
)

add_entrypoint_object(
  gmtime
  SRCS
    gmtime.cpp
  HDRS
    gmtime.h
  DEPENDS
    .time_utils
    libc.include.time
    libc.hdr.types.time_t
    libc.hdr.types.struct_tm
)

add_entrypoint_object(
  gmtime_r
  SRCS
    gmtime_r.cpp
  HDRS
    gmtime_r.h
  DEPENDS
    .time_utils
    libc.include.time
    libc.hdr.types.time_t
    libc.hdr.types.struct_tm
)

add_entrypoint_object(
  mktime
  SRCS
    mktime.cpp
  HDRS
    mktime.h
  DEPENDS
    .time_utils
    .time_constants
    libc.include.time
    libc.src.errno.errno
    libc.hdr.types.time_t
    libc.hdr.types.struct_tm
)

add_subdirectory(strftime_core) #TODO: Move to top

add_entrypoint_object(
  strftime
  SRCS
    strftime.cpp
  HDRS
    strftime.h
  DEPENDS
    libc.hdr.types.size_t
    libc.hdr.types.struct_tm
    libc.src.stdio.printf_core.writer
    libc.src.time.strftime_core.strftime_main
)

add_entrypoint_object(
  strftime_l
  SRCS
    strftime_l.cpp
  HDRS
    strftime_l.h
  DEPENDS
    libc.hdr.types.locale_t
    libc.hdr.types.size_t
    libc.hdr.types.struct_tm
    libc.src.stdio.printf_core.writer
    libc.src.time.strftime_core.strftime_main
)

add_entrypoint_object(
  time
  SRCS
    time.cpp
  HDRS
    time_func.h
  DEPENDS
    libc.hdr.time_macros
    libc.hdr.types.time_t
    libc.src.__support.time.clock_gettime
    libc.src.errno.errno
    libc.hdr.types.struct_tm
)

add_entrypoint_object(
  timespec_get
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.timespec_get
)

add_entrypoint_object(
  clock
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.clock
)

add_entrypoint_object(
  nanosleep
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.nanosleep
)

add_entrypoint_object(
  clock_gettime
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.clock_gettime
)

add_entrypoint_object(
  gettimeofday
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.gettimeofday
)

add_entrypoint_object(
  clock_getres
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.clock_getres
)
