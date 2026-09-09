if(NOT (TARGET libc.src.__support.threads.mutex) OR LIBC_TARGET_OS_IS_GPU)
  return()
endif()

add_libc_test(
  field_tokenizer_test
  SUITE
    libc-support-tests
  SRCS
    field_tokenizer_test.cpp
  DEPENDS
    libc.src.__support.CPP.span
    libc.src.__support.CPP.string_view
    libc.src.__support.pwd.field_tokenizer
)

add_libc_test(
  flat_file_db_test
  SUITE
    libc-support-tests
  SRCS
    flat_file_db_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.types.size_t
    libc.src.__support.CPP.span
    libc.src.__support.CPP.string_view
    libc.src.__support.File.file
    libc.src.__support.File.platform_file
    libc.src.__support.pwd.flat_file_db
    libc.src.stdio.remove
    libc.src.string.string_utils
)
