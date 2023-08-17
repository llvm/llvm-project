if(LIBC_TARGET_ARCHITECTURE_IS_GPU)
  # The GPU build cannot use any generic implementations.
  return()
endif()

add_entrypoint_object(
  fputs
  SRCS
    fputs.cpp
  HDRS
    ../fputs.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  puts
  SRCS
    puts.cpp
  HDRS
    ../puts.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  stdin
  SRCS
    stdin.cpp
  HDRS
    ../stdin.h
  DEPENDS
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  stdout
  SRCS
    stdout.cpp
  HDRS
    ../stdout.h
  DEPENDS
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  stderr
  SRCS
    stderr.cpp
  HDRS
    ../stderr.h
  DEPENDS
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)
