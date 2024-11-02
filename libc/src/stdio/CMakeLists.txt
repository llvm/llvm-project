# Helper function that creates an alias if a target specific implementation
# exists, otherwise it uses a generic one.
function(add_stdio_entrypoint_object name)
  if(TARGET libc.src.stdio.${LIBC_TARGET_OS}.${name})
    add_entrypoint_object(
      ${name}
      ALIAS
      DEPENDS
        .${LIBC_TARGET_OS}.${name}
    )
  elseif(TARGET libc.src.stdio.generic.${name})
    add_entrypoint_object(
      ${name}
      ALIAS
      DEPENDS
        .generic.${name}
    )
  endif()
endfunction(add_stdio_entrypoint_object)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

if(NOT LIBC_TARGET_OS_IS_BAREMETAL AND NOT LIBC_TARGET_OS_IS_GPU)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/generic)
endif()

if(NOT LLVM_LIBC_FULL_BUILD)
  list(APPEND use_system_file "COMPILE_OPTIONS" "-DLIBC_COPT_STDIO_USE_SYSTEM_FILE")
endif()

add_entrypoint_object(
  flockfile
  SRCS
    flockfile.cpp
  HDRS
    flockfile.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  funlockfile
  SRCS
    funlockfile.cpp
  HDRS
    funlockfile.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  fopencookie
  SRCS
    fopencookie.cpp
  HDRS
    fopencookie.h
  DEPENDS
    libc.hdr.stdio_macros
    libc.hdr.types.off_t
    libc.hdr.types.cookie_io_functions_t
    libc.hdr.types.FILE
    libc.src.__support.CPP.new
    libc.src.__support.File.file
)

add_entrypoint_object(
  setbuf
  SRCS
    setbuf.cpp
  HDRS
    setbuf.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.off_t
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  setvbuf
  SRCS
    setvbuf.cpp
  HDRS
    setvbuf.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

list(APPEND scanf_deps
      libc.src.__support.arg_list
      libc.src.stdio.scanf_core.vfscanf_internal
      libc.hdr.types.FILE
)

if(LLVM_LIBC_FULL_BUILD AND NOT LIBC_TARGET_OS_IS_GPU)
  list(APPEND scanf_deps
      libc.src.__support.File.file
      libc.src.__support.File.platform_file
      libc.src.__support.File.platform_stdin
  )
endif()

add_entrypoint_object(
  sscanf
  SRCS
    sscanf.cpp
  HDRS
    sscanf.h
  DEPENDS
    libc.src.__support.arg_list
    libc.src.stdio.scanf_core.reader
    libc.src.stdio.scanf_core.scanf_main
)

add_entrypoint_object(
  vsscanf
  SRCS
    vsscanf.cpp
  HDRS
    vsscanf.h
  DEPENDS
    libc.src.__support.arg_list
    libc.src.stdio.scanf_core.reader
    libc.src.stdio.scanf_core.scanf_main
)

add_entrypoint_object(
  fscanf
  SRCS
    fscanf.cpp
  HDRS
    fscanf.h
  DEPENDS
    ${scanf_deps}
)

add_entrypoint_object(
  vfscanf
  SRCS
    vfscanf.cpp
  HDRS
    vfscanf.h
  DEPENDS
    ${scanf_deps}
)

add_entrypoint_object(
  scanf
  SRCS
    scanf.cpp
  HDRS
    scanf.h
  DEPENDS
    ${scanf_deps}
)

add_entrypoint_object(
  vscanf
  SRCS
    vscanf.cpp
  HDRS
    vscanf.h
  DEPENDS
    ${scanf_deps}
)

add_entrypoint_object(
  sprintf
  SRCS
    sprintf.cpp
  HDRS
    sprintf.h
  DEPENDS
    libc.src.stdio.printf_core.printf_main
    libc.src.stdio.printf_core.writer
)

add_entrypoint_object(
  snprintf
  SRCS
    snprintf.cpp
  HDRS
    snprintf.h
  DEPENDS
    libc.src.stdio.printf_core.printf_main
    libc.src.stdio.printf_core.writer
)

add_entrypoint_object(
  asprintf
  SRCS
    asprintf.cpp
  HDRS
    asprintf.h
  DEPENDS
    libc.src.stdio.printf_core.vasprintf_internal
)

add_entrypoint_object(
  vsprintf
  SRCS
    vsprintf.cpp
  HDRS
    vsprintf.h
  DEPENDS
    libc.src.stdio.printf_core.printf_main
    libc.src.stdio.printf_core.writer
)

add_entrypoint_object(
  vsnprintf
  SRCS
    vsnprintf.cpp
  HDRS
    vsnprintf.h
  DEPENDS
    libc.src.stdio.printf_core.printf_main
    libc.src.stdio.printf_core.writer
)

add_entrypoint_object(
  vasprintf
  SRCS
    vasprintf.cpp
  HDRS
    vasprintf.h
  DEPENDS
    libc.src.stdio.printf_core.vasprintf_internal
)

add_subdirectory(printf_core)
add_subdirectory(scanf_core)

add_entrypoint_object(
  remove
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.remove
)

add_entrypoint_object(
  rename
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.rename
)

add_entrypoint_object(
  fdopen
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.fdopen
)

# These entrypoints have multiple potential implementations.
add_stdio_entrypoint_object(feof)
add_stdio_entrypoint_object(feof_unlocked)
add_stdio_entrypoint_object(ferror)
add_stdio_entrypoint_object(ferror_unlocked)
add_stdio_entrypoint_object(fseek)
add_stdio_entrypoint_object(ftell)
add_stdio_entrypoint_object(fseeko)
add_stdio_entrypoint_object(fileno)
add_stdio_entrypoint_object(ftello)
add_stdio_entrypoint_object(fflush)
add_stdio_entrypoint_object(clearerr)
add_stdio_entrypoint_object(clearerr_unlocked)
add_stdio_entrypoint_object(fopen)
add_stdio_entrypoint_object(fclose)
add_stdio_entrypoint_object(fread_unlocked)
add_stdio_entrypoint_object(fread)
add_stdio_entrypoint_object(puts)
add_stdio_entrypoint_object(fputs)
add_stdio_entrypoint_object(fwrite_unlocked)
add_stdio_entrypoint_object(fwrite)
add_stdio_entrypoint_object(fputc)
add_stdio_entrypoint_object(putc)
add_stdio_entrypoint_object(putchar)
add_stdio_entrypoint_object(printf)
add_stdio_entrypoint_object(fprintf)
add_stdio_entrypoint_object(fgetc)
add_stdio_entrypoint_object(fgetc_unlocked)
add_stdio_entrypoint_object(getc)
add_stdio_entrypoint_object(getc_unlocked)
add_stdio_entrypoint_object(getchar)
add_stdio_entrypoint_object(getchar_unlocked)
add_stdio_entrypoint_object(fgets)
add_stdio_entrypoint_object(ungetc)
add_stdio_entrypoint_object(stdin)
add_stdio_entrypoint_object(stdout)
add_stdio_entrypoint_object(stderr)
add_stdio_entrypoint_object(vprintf)
add_stdio_entrypoint_object(vfprintf)
