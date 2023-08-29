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

if(NOT LIBC_TARGET_ARCHITECTURE_IS_GPU)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/generic)
endif()

add_subdirectory(printf_core)
add_subdirectory(scanf_core)

add_entrypoint_object(
  clearerr
  SRCS
    clearerr.cpp
  HDRS
    clearerr.h
  DEPENDS
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  clearerr_unlocked
  SRCS
    clearerr_unlocked.cpp
  HDRS
    clearerr_unlocked.h
  DEPENDS
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  feof
  SRCS
    feof.cpp
  HDRS
    feof.h
  DEPENDS
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  feof_unlocked
  SRCS
    feof_unlocked.cpp
  HDRS
    feof_unlocked.h
  DEPENDS
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  ferror
  SRCS
    ferror.cpp
  HDRS
    ferror.h
  DEPENDS
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  ferror_unlocked
  SRCS
    ferror_unlocked.cpp
  HDRS
    ferror_unlocked.h
  DEPENDS
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  fgetc
  SRCS
    fgetc.cpp
  HDRS
    fgetc.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  fgetc_unlocked
  SRCS
    fgetc_unlocked.cpp
  HDRS
    fgetc_unlocked.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  getc
  SRCS
    getc.cpp
  HDRS
    getc.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  getc_unlocked
  SRCS
    getc_unlocked.cpp
  HDRS
    getc_unlocked.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  getchar
  SRCS
    getchar.cpp
  HDRS
    getchar.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  getchar_unlocked
  SRCS
    getc_unlocked.cpp
  HDRS
    getc_unlocked.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  fgets
  SRCS
    fgets.cpp
  HDRS
    fgets.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  fflush
  SRCS
    fflush.cpp
  HDRS
    fflush.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  flockfile
  SRCS
    flockfile.cpp
  HDRS
    flockfile.h
  DEPENDS
    libc.include.stdio
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
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  fwrite_unlocked
  SRCS
    fwrite_unlocked.cpp
  HDRS
    fwrite_unlocked.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  fwrite
  SRCS
    fwrite.cpp
  HDRS
    fwrite.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  fputc
  SRCS
    fputc.cpp
  HDRS
    fputc.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  putc
  SRCS
    putc.cpp
  HDRS
    putc.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  putchar
  SRCS
    putchar.cpp
  HDRS
    putchar.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  fseek
  SRCS
    fseek.cpp
  HDRS
    fseek.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  ungetc
  SRCS
    ungetc.cpp
  HDRS
    ungetc.h
  DEPENDS
    libc.include.stdio
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
    libc.include.stdio
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
    libc.include.stdio
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
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  sscanf
  SRCS
    sscanf.cpp
  HDRS
    sscanf.h
  DEPENDS
    libc.src.__support.arg_list
    libc.src.stdio.scanf_core.string_reader
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
    libc.src.__support.arg_list
    libc.src.stdio.scanf_core.vfscanf_internal
)

add_entrypoint_object(
  scanf
  SRCS
    scanf.cpp
  HDRS
    scanf.h
  DEPENDS
    libc.src.__support.arg_list
    libc.src.stdio.scanf_core.vfscanf_internal
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

list(APPEND printf_deps
      libc.src.__support.arg_list
      libc.src.stdio.printf_core.vfprintf_internal
)

if(LIBC_CONF_PRINTF_DISABLE_FLOAT)
  list(APPEND printf_copts "-DLIBC_COPT_PRINTF_DISABLE_FLOAT")
endif()
if(LIBC_CONF_PRINTF_DISABLE_INDEX_MODE)
  list(APPEND printf_copts "-DLIBC_COPT_PRINTF_DISABLE_INDEX_MODE")
endif()
if(LIBC_CONF_PRINTF_DISABLE_WRITE_INT)
  list(APPEND printf_copts "-DLIBC_COPT_PRINTF_DISABLE_WRITE_INT")
endif()

if(LLVM_LIBC_FULL_BUILD)
  list(APPEND printf_deps
      libc.src.__support.File.file
      libc.src.__support.File.platform_file
      libc.src.__support.File.platform_stdout
  )
else()
  list(APPEND printf_copts "-DLIBC_COPT_PRINTF_USE_SYSTEM_FILE")
endif()

add_entrypoint_object(
  printf
  SRCS
    printf.cpp
  HDRS
    printf.h
  DEPENDS
    ${printf_deps}
  COMPILE_OPTIONS
    ${printf_copts}
)

add_entrypoint_object(
  fprintf
  SRCS
    fprintf.cpp
  HDRS
    fprintf.h
  DEPENDS
    libc.src.__support.arg_list
    libc.src.stdio.printf_core.vfprintf_internal
  COMPILE_OPTIONS
    ${printf_copts}
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
  vprintf
  SRCS
    vprintf.cpp
  HDRS
    vprintf.h
  DEPENDS
    ${printf_deps}
  COMPILE_OPTIONS
    ${printf_copts}
)

add_entrypoint_object(
  vfprintf
  SRCS
    vfprintf.cpp
  HDRS
    vfprintf.h
  DEPENDS
    libc.src.__support.arg_list
    libc.src.stdio.printf_core.vfprintf_internal
  COMPILE_OPTIONS
    ${printf_copts}
)

add_entrypoint_object(
  ftell
  SRCS
    ftell.cpp
  HDRS
    ftell.h
  DEPENDS
    libc.src.errno.errno
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_entrypoint_object(
  remove
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.remove
)

# These entrypoints have multiple potential implementations.
add_stdio_entrypoint_object(fopen)
add_stdio_entrypoint_object(fclose)
add_stdio_entrypoint_object(fread_unlocked)
add_stdio_entrypoint_object(fread)
add_stdio_entrypoint_object(puts)
add_stdio_entrypoint_object(fputs)
add_stdio_entrypoint_object(stdin)
add_stdio_entrypoint_object(stdout)
add_stdio_entrypoint_object(stderr)
