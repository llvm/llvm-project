# Helper to only add the generic implementations if they aren't handled by a
# more specific implementation.
# TODO: This should probably be cleaned up and formalized.
function(add_generic_entrypoint_object name)
  if(NOT TARGET libc.src.stdio.${LIBC_TARGET_OS}.${name})
    add_entrypoint_object(
      ${name}
      ${ARGN}
    )
  endif()
endfunction(add_generic_entrypoint_object)

add_generic_entrypoint_object(
  clearerr
  SRCS
    clearerr.cpp
  HDRS
    ../clearerr.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  clearerr_unlocked
  SRCS
    clearerr_unlocked.cpp
  HDRS
    ../clearerr_unlocked.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  feof
  SRCS
    feof.cpp
  HDRS
    ../feof.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  feof_unlocked
  SRCS
    feof_unlocked.cpp
  HDRS
    ../feof_unlocked.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  ferror
  SRCS
    ferror.cpp
  HDRS
    ../ferror.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  ferror_unlocked
  SRCS
    ferror_unlocked.cpp
  HDRS
    ../ferror_unlocked.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  fileno
  SRCS
    fileno.cpp
  HDRS
    ../fileno.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  fflush
  SRCS
    fflush.cpp
  HDRS
    ../fflush.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  fseek
  SRCS
    fseek.cpp
  HDRS
    ../fseek.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  ftell
  SRCS
    ftell.cpp
  HDRS
    ../ftell.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  fseeko
  SRCS
    fseeko.cpp
  HDRS
    ../fseeko.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  ftello
  SRCS
    ftello.cpp
  HDRS
    ../ftello.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  fopen
  SRCS
    fopen.cpp
  HDRS
    ../fopen.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  fclose
  SRCS
    fclose.cpp
  HDRS
    ../fclose.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.errno.errno
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  fread_unlocked
  SRCS
    fread_unlocked.cpp
  HDRS
    ../fread_unlocked.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  fread
  SRCS
    fread.cpp
  HDRS
    ../fread.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  perror
  SRCS
    perror.cpp
  HDRS
    ../perror.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.StringUtil.error_to_string
    libc.src.__support.CPP.string_view
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
    libc.src.__support.File.platform_stderr
)

add_generic_entrypoint_object(
  fputs
  SRCS
    fputs.cpp
  HDRS
    ../fputs.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  puts
  SRCS
    puts.cpp
  HDRS
    ../puts.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_stdout
)

add_generic_entrypoint_object(
  fwrite_unlocked
  SRCS
    fwrite_unlocked.cpp
  HDRS
    ../fwrite_unlocked.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  fwrite
  SRCS
    fwrite.cpp
  HDRS
    ../fwrite.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  fputc
  SRCS
    fputc.cpp
  HDRS
    ../fputc.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  putc
  SRCS
    putc.cpp
  HDRS
    ../putc.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  putchar
  SRCS
    putchar.cpp
  HDRS
    ../putchar.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  fgetc
  SRCS
    fgetc.cpp
  HDRS
    ../fgetc.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  fgetc_unlocked
  SRCS
    fgetc_unlocked.cpp
  HDRS
    ../fgetc_unlocked.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  getc
  SRCS
    getc.cpp
  HDRS
    ../getc.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  getc_unlocked
  SRCS
    getc_unlocked.cpp
  HDRS
    ../getc_unlocked.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  getchar
  SRCS
    getchar.cpp
  HDRS
    ../getchar.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  getchar_unlocked
  SRCS
    getchar_unlocked.cpp
  HDRS
    ../getchar_unlocked.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

list(APPEND fprintf_deps
      libc.hdr.types.FILE
      libc.src.__support.arg_list
      libc.src.stdio.printf_core.vfprintf_internal
)

if(LLVM_LIBC_FULL_BUILD)
  list(APPEND fprintf_deps
      libc.src.__support.File.file
      libc.src.__support.File.platform_file
  )
endif()

# Copy the deps for printf_deps
set(printf_deps ${fprintf_deps})

if(LLVM_LIBC_FULL_BUILD)
  list(APPEND printf_deps
      libc.src.__support.File.platform_stdout
  )
endif()

add_generic_entrypoint_object(
  printf
  SRCS
    printf.cpp
  HDRS
    ../printf.h
  DEPENDS
    ${printf_deps}
)

add_generic_entrypoint_object(
  vprintf
  SRCS
    vprintf.cpp
  HDRS
    ../vprintf.h
  DEPENDS
    ${printf_deps}
)

add_generic_entrypoint_object(
  fprintf
  SRCS
    fprintf.cpp
  HDRS
    ../fprintf.h
  DEPENDS
    ${fprintf_deps}
)

add_generic_entrypoint_object(
  vfprintf
  SRCS
    vfprintf.cpp
  HDRS
    ../vfprintf.h
  DEPENDS
    ${fprintf_deps}
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

add_generic_entrypoint_object(
  fscanf
  SRCS
    fscanf.cpp
  HDRS
    ../fscanf.h
  DEPENDS
    ${scanf_deps}
)

add_generic_entrypoint_object(
  vfscanf
  SRCS
    vfscanf.cpp
  HDRS
    ../vfscanf.h
  DEPENDS
    ${scanf_deps}
)

add_generic_entrypoint_object(
  scanf
  SRCS
    scanf.cpp
  HDRS
    ../scanf.h
  DEPENDS
    ${scanf_deps}
)

add_generic_entrypoint_object(
  vscanf
  SRCS
    vscanf.cpp
  HDRS
    ../vscanf.h
  DEPENDS
    ${scanf_deps}
)

add_generic_entrypoint_object(
  fgets
  SRCS
    fgets.cpp
  HDRS
    ../fgets.h
  DEPENDS
    libc.src.errno.errno
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  ungetc
  SRCS
    ungetc.cpp
  HDRS
    ../ungetc.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
)

add_generic_entrypoint_object(
  stdin
  SRCS
    stdin.cpp
  HDRS
    ../stdin.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_stdin
)

add_generic_entrypoint_object(
  stdout
  SRCS
    stdout.cpp
  HDRS
    ../stdout.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_stdout
)

add_generic_entrypoint_object(
  stderr
  SRCS
    stderr.cpp
  HDRS
    ../stderr.h
  DEPENDS
    libc.hdr.types.FILE
    libc.src.__support.File.file
    libc.src.__support.File.platform_stderr
)
