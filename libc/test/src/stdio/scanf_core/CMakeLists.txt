if(NOT(LLVM_LIBC_FULL_BUILD))
  # in overlay mode, use the system's file to test the reader.
  set(use_system_file "-DLIBC_COPT_STDIO_USE_SYSTEM_FILE")
endif()

add_libc_unittest(
  parser_test
  SUITE
    libc_stdio_unittests
  SRCS
    parser_test.cpp
  LINK_LIBRARIES
    LibcScanfHelpers
  DEPENDS
    libc.src.stdio.scanf_core.parser
    libc.src.stdio.scanf_core.core_structs
    libc.src.__support.CPP.string_view
    libc.src.__support.arg_list
)

if(NOT(TARGET libc.src.__support.File.file) AND LLVM_LIBC_FULL_BUILD AND
  (NOT LIBC_TARGET_OS_IS_GPU))
  # Not all platforms have a file implementation. If file is unvailable, and a
  # full build is requested, then we must skip all file based scanf sections.
  return()
endif()

add_libc_unittest(
  reader_test
  SUITE
    libc_stdio_unittests
  SRCS
    reader_test.cpp
  DEPENDS
    libc.src.stdio.scanf_core.string_reader
    libc.src.__support.CPP.string_view
  COMPILE_OPTIONS
    ${use_system_file}
)

add_libc_unittest(
  converter_test
  SUITE
    libc_stdio_unittests
  SRCS
    converter_test.cpp
  DEPENDS
    libc.src.stdio.scanf_core.converter
    libc.src.stdio.scanf_core.string_reader
    libc.src.__support.CPP.string_view
  COMPILE_OPTIONS
    ${use_system_file}
)
