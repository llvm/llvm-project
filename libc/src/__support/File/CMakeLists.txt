if(NOT (TARGET libc.src.__support.threads.mutex)
   OR LIBC_TARGET_OS_IS_GPU)
  # Not all platforms have a mutex implementation. If mutex is unvailable,
  # we just skip everything about files.
  return()
endif()

add_object_library(
  file
  SRCS
    file.cpp
  HDRS
    file.h
  DEPENDS
    libc.hdr.stdio_macros
    libc.hdr.stdint_proxy
    libc.hdr.func.realloc
    libc.hdr.types.off_t
    libc.src.__support.CPP.new
    libc.src.__support.CPP.span
    libc.src.__support.threads.mutex
    libc.src.__support.error_or
)

add_object_library(
  dir
  SRCS
    dir.cpp
  HDRS
    dir.h
  DEPENDS
    libc.src.__support.CPP.mutex
    libc.src.__support.CPP.new
    libc.src.__support.CPP.span
    libc.src.__support.threads.mutex
)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  return()
endif()

add_subdirectory(${LIBC_TARGET_OS})

set(target_file libc.src.__support.File.${LIBC_TARGET_OS}.file)
set(target_stdout libc.src.__support.File.${LIBC_TARGET_OS}.stdout)
set(target_stderr libc.src.__support.File.${LIBC_TARGET_OS}.stderr)
set(target_stdin libc.src.__support.File.${LIBC_TARGET_OS}.stdin)

set(file_targets "${target_file};${target_stdout};${target_stdin};${target_stderr}")
set(file_aliases "platform_file;platform_stdout;platform_stdin;platform_stderr")

foreach(alias target IN ZIP_LISTS file_aliases file_targets)
  if(TARGET ${target})
    add_object_library(
      ${alias}
      ALIAS
        ${target}
      DEPENDS
        ${target}
    )
  endif()
endforeach()

set(target_dir libc.src.__support.File.${LIBC_TARGET_OS}.${LIBC_TARGET_OS}_dir)
if(NOT TARGET ${target_dir})
  return()
endif()

add_object_library(
  platform_dir
  ALIAS
    ${target_dir}
  DEPENDS
    ${target_dir}
)
