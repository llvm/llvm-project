add_custom_target(libc_stdio_unittests)

add_libc_test(
  fileop_test
  SUITE
    libc_stdio_unittests
  SRCS
    fileop_test.cpp
  DEPENDS
    libc.include.stdio
    libc.src.errno.errno
    libc.src.stdio.clearerr
    libc.src.stdio.fclose
    libc.src.stdio.feof
    libc.src.stdio.ferror
    libc.src.stdio.fflush
    libc.src.stdio.fileno
    libc.src.stdio.fopen
    libc.src.stdio.fputs
    libc.src.stdio.fread
    libc.src.stdio.fseek
    libc.src.stdio.fwrite
)

add_libc_test(
  ungetc_test
  SUITE
    libc_stdio_unittests
  SRCS
    ungetc_test.cpp
  DEPENDS
    libc.include.stdio
    libc.src.stdio.fclose
    libc.src.stdio.fopen
    libc.src.stdio.fread
    libc.src.stdio.fseek
    libc.src.stdio.fwrite
    libc.src.stdio.ungetc
)

add_libc_test(
  setbuf_test
  SUITE
    libc_stdio_unittests
  SRCS
    setbuf_test.cpp
  DEPENDS
    libc.include.stdio
    libc.src.stdio.fclose
    libc.src.stdio.fopen
    libc.src.stdio.fread
    libc.src.stdio.fwrite
    libc.src.stdio.setbuf
    libc.src.stdio.ungetc
)

add_libc_test(
  setvbuf_test
  SUITE
    libc_stdio_unittests
  SRCS
    setvbuf_test.cpp
  DEPENDS
    libc.include.stdio
    libc.src.errno.errno
    libc.src.stdio.fclose
    libc.src.stdio.fopen
    libc.src.stdio.fread
    libc.src.stdio.fwrite
    libc.src.stdio.setvbuf
)

add_libc_test(
  unlocked_fileop_test
  SUITE
    libc_stdio_unittests
  SRCS
    unlocked_fileop_test.cpp
  DEPENDS
    libc.include.stdio
    libc.src.errno.errno
    libc.src.stdio.clearerr_unlocked
    libc.src.stdio.fclose
    libc.src.stdio.feof_unlocked
    libc.src.stdio.ferror_unlocked
    libc.src.stdio.flockfile
    libc.src.stdio.fopen
    libc.src.stdio.fread_unlocked
    libc.src.stdio.funlockfile
    libc.src.stdio.fwrite_unlocked
)

add_libc_test(
  fopencookie_test
  SUITE
    libc_stdio_unittests
  SRCS
    fopencookie_test.cpp
  DEPENDS
    libc.include.stdio
    libc.include.stdlib
    libc.src.errno.errno
    libc.src.stdio.clearerr
    libc.src.stdio.fclose
    libc.src.stdio.feof
    libc.src.stdio.ferror
    libc.src.stdio.fflush
    libc.src.stdio.fopencookie
    libc.src.stdio.fread
    libc.src.stdio.fseek
    libc.src.stdio.fwrite
  LINK_LIBRARIES
    LibcMemoryHelpers
)

if(LIBC_CONF_PRINTF_DISABLE_FLOAT)
  list(APPEND sprintf_test_copts "-DLIBC_COPT_PRINTF_DISABLE_FLOAT")
endif()
if(LIBC_CONF_PRINTF_DISABLE_INDEX_MODE)
  list(APPEND sprintf_test_copts "-DLIBC_COPT_PRINTF_DISABLE_INDEX_MODE")
endif()
if(LIBC_CONF_PRINTF_DISABLE_WRITE_INT)
  list(APPEND sprintf_test_copts "-DLIBC_COPT_PRINTF_DISABLE_WRITE_INT")
endif()
if(LIBC_CONF_PRINTF_DISABLE_FIXED_POINT)
  list(APPEND sprintf_test_copts "-DLIBC_COPT_PRINTF_DISABLE_FIXED_POINT")
endif()

add_fp_unittest(
  sprintf_test
  UNIT_TEST_ONLY
  SUITE
    libc_stdio_unittests
  SRCS
    sprintf_test.cpp
  DEPENDS
    libc.src.stdio.sprintf
    libc.src.__support.FPUtil.fp_bits
    libc.include.inttypes
  COMPILE_OPTIONS
    ${sprintf_test_copts}
)

add_libc_test(
  snprintf_test
  SUITE
    libc_stdio_unittests
  SRCS
    snprintf_test.cpp
  DEPENDS
    libc.src.stdio.snprintf
)

if(LLVM_LIBC_FULL_BUILD)
  # In fullbuild mode, fprintf's tests use the internal FILE for other functions.
  list(APPEND fprintf_test_deps
       libc.src.stdio.fclose
       libc.src.stdio.ferror
       libc.src.stdio.fopen
       libc.src.stdio.fread
  )
  # This is to be used for tests which write to libc's platform streams
  # under full build but write to system-lib's streams otherwise.
  set(hermetic_test_only HERMETIC_TEST_ONLY)
else()
# Else in overlay mode they use the system's FILE.
 set(use_system_file "-DLIBC_COPT_STDIO_USE_SYSTEM_FILE")
endif()

add_libc_test(
  fprintf_test
  SUITE
    libc_stdio_unittests
  SRCS
    fprintf_test.cpp
  DEPENDS
    libc.src.stdio.fprintf
    ${fprintf_test_deps}
  COMPILE_OPTIONS
    ${use_system_file}
)

add_libc_test(
  printf_test
  ${hermetic_test_only}
  SUITE
    libc_stdio_unittests
  SRCS
    printf_test.cpp
  DEPENDS
    libc.src.stdio.printf
)

add_fp_unittest(
  vsprintf_test
  UNIT_TEST_ONLY
  SUITE
    libc_stdio_unittests
  SRCS
    vsprintf_test.cpp
  DEPENDS
    libc.src.stdio.vsprintf
)

add_libc_test(
  vsnprintf_test
  SUITE
    libc_stdio_unittests
  SRCS
    vsnprintf_test.cpp
  DEPENDS
    libc.src.stdio.vsnprintf
)

add_libc_test(
  vfprintf_test
  SUITE
    libc_stdio_unittests
  SRCS
    vfprintf_test.cpp
  DEPENDS
    libc.src.stdio.vfprintf
    ${fprintf_test_deps}
  COMPILE_OPTIONS
    ${use_system_file}
)

add_libc_test(
  vprintf_test
  ${hermetic_test_only}
  SUITE
    libc_stdio_unittests
  SRCS
    vprintf_test.cpp
  DEPENDS
    libc.src.stdio.vprintf
)


if(LLVM_LIBC_FULL_BUILD)
  # In fullbuild mode, fscanf's tests use the internal FILE for other functions.
  list(APPEND fscanf_test_deps
       libc.src.stdio.fclose
       libc.src.stdio.ferror
       libc.src.stdio.fopen
       libc.src.stdio.fwrite
  )
endif()

add_libc_test(
  fscanf_test
  SUITE
    libc_stdio_unittests
  SRCS
    fscanf_test.cpp
  DEPENDS
    libc.src.stdio.fscanf
    ${fscanf_test_deps}
    libc.src.__support.CPP.string_view
  COMPILE_OPTIONS
    ${use_system_file}
)

add_libc_test(
  sscanf_test
  SUITE
    libc_stdio_unittests
  SRCS
    sscanf_test.cpp
  DEPENDS
    libc.src.stdio.sscanf
  LINK_LIBRARIES
    LibcFPTestHelpers
)

add_libc_test(
  puts_test
  HERMETIC_TEST_ONLY # writes to libc's stdout
  SUITE
    libc_stdio_unittests
  SRCS
    puts_test.cpp
  DEPENDS
    libc.src.stdio.puts
)

add_libc_test(
  fputs_test
  HERMETIC_TEST_ONLY # writes to libc's stdout and stderr
  SUITE
    libc_stdio_unittests
  SRCS
    fputs_test.cpp
  DEPENDS
    libc.src.stdio.fputs
    libc.src.stdio.stdout
    libc.src.stdio.stderr
)

add_libc_test(
  fputc_test
  HERMETIC_TEST_ONLY # writes to libc's stdout and stderr
  SUITE
    libc_stdio_unittests
  SRCS
    fputc_test.cpp
  DEPENDS
    libc.src.stdio.fputc
    libc.src.stdio.putchar
    libc.src.stdio.stdout
    libc.src.stdio.stderr
)

add_libc_test(
  fopen_test
  SUITE
    libc_stdio_unittests
  SRCS
    fopen_test.cpp
  DEPENDS
    libc.src.stdio.fread
    libc.src.stdio.fwrite
    libc.src.stdio.fclose
    libc.src.stdio.fopen
)

add_libc_test(
  putc_test
  SUITE
    libc_stdio_unittests
  SRCS
    putc_test.cpp
  DEPENDS
    libc.src.stdio.putc
    libc.src.stdio.fclose
    libc.src.stdio.ferror
    libc.src.stdio.fopen
    libc.src.stdio.fread
)

if(${LIBC_TARGET_OS} STREQUAL "linux")
  add_libc_test(
    remove_test
    SUITE
      libc_stdio_unittests
    SRCS
      remove_test.cpp
    DEPENDS
      libc.include.unistd
      libc.src.errno.errno
      libc.src.fcntl.open
      libc.src.stdio.remove
      libc.src.sys.stat.mkdirat
      libc.src.unistd.access
      libc.src.unistd.close
  )

  add_libc_test(
    rename_test
    SUITE
      libc_stdio_unittests
    SRCS
      rename_test.cpp
    DEPENDS
      libc.src.errno.errno
      libc.src.fcntl.open
      libc.src.stdio.rename
      libc.src.unistd.access
      libc.src.unistd.close
      libc.test.UnitTest.ErrnoSetterMatcher
  )

  add_libc_test(
    fdopen_test
    SUITE
      libc_stdio_unittests
    SRCS
      fdopen_test.cpp
    DEPENDS
      libc.src.fcntl.open
      libc.src.stdio.fclose
      libc.src.stdio.fdopen
      libc.src.stdio.fgets
      libc.src.stdio.fputs
      libc.src.unistd.close
      libc.test.UnitTest.ErrnoSetterMatcher
  )
endif()

add_libc_test(
  fgetc_test
  SUITE
    libc_stdio_unittests
  SRCS
    fgetc_test.cpp
  DEPENDS
    libc.include.stdio
    libc.src.errno.errno
    libc.src.stdio.fclose
    libc.src.stdio.feof
    libc.src.stdio.ferror
    libc.src.stdio.fgetc
    libc.src.stdio.fopen
    libc.src.stdio.fwrite
    libc.src.stdio.getc
)

add_libc_test(
  fgetc_unlocked_test
  SUITE
    libc_stdio_unittests
  SRCS
    fgetc_unlocked_test.cpp
  DEPENDS
    libc.include.stdio
    libc.src.errno.errno
    libc.src.stdio.fclose
    libc.src.stdio.ferror
    libc.src.stdio.ferror_unlocked
    libc.src.stdio.feof
    libc.src.stdio.feof_unlocked
    libc.src.stdio.fgetc_unlocked
    libc.src.stdio.flockfile
    libc.src.stdio.fopen
    libc.src.stdio.funlockfile
    libc.src.stdio.fwrite
    libc.src.stdio.getc_unlocked
)

add_libc_test(
  fgets_test
  SUITE
    libc_stdio_unittests
  SRCS
    fgets_test.cpp
  DEPENDS
    libc.include.stdio
    libc.src.errno.errno
    libc.src.stdio.fclose
    libc.src.stdio.feof
    libc.src.stdio.ferror
    libc.src.stdio.fgets
    libc.src.stdio.fopen
    libc.src.stdio.fwrite
)

add_libc_test(
  ftell_test
  SUITE
    libc_stdio_unittests
  SRCS
    ftell_test.cpp
  DEPENDS
    libc.include.stdio
    libc.src.stdio.fclose
    libc.src.stdio.fflush
    libc.src.stdio.fopen
    libc.src.stdio.fread
    libc.src.stdio.fseek
    libc.src.stdio.ftell
    libc.src.stdio.fseeko
    libc.src.stdio.ftello
    libc.src.stdio.fwrite
    libc.src.stdio.setvbuf
)

# Create an output directory for any temporary test files.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testdata)

if(LIBC_TARGET_OS_IS_GPU)
  return()
endif()

add_subdirectory(printf_core)
add_subdirectory(scanf_core)
add_subdirectory(testdata)
