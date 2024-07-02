if(LIBC_CONF_PRINTF_DISABLE_FLOAT)
  list(APPEND printf_config_copts "-DLIBC_COPT_PRINTF_DISABLE_FLOAT")
endif()
if(LIBC_CONF_PRINTF_DISABLE_INDEX_MODE)
  list(APPEND printf_config_copts "-DLIBC_COPT_PRINTF_DISABLE_INDEX_MODE")
endif()
if(LIBC_CONF_PRINTF_DISABLE_WRITE_INT)
  list(APPEND printf_config_copts "-DLIBC_COPT_PRINTF_DISABLE_WRITE_INT")
endif()
if(LIBC_CONF_PRINTF_FLOAT_TO_STR_USE_MEGA_LONG_DOUBLE_TABLE)
  list(APPEND printf_config_copts "-DLIBC_COPT_FLOAT_TO_STR_USE_MEGA_LONG_DOUBLE_TABLE")
endif()
if(LIBC_CONF_PRINTF_DISABLE_FIXED_POINT)
  list(APPEND printf_config_copts "-DLIBC_COPT_PRINTF_DISABLE_FIXED_POINT")
endif()
if(printf_config_copts)
  list(PREPEND printf_config_copts "COMPILE_OPTIONS")
endif()

add_header_library(
  printf_config
  HDRS
    printf_config.h
  ${printf_config_copts}
)

add_header_library(
  core_structs
  HDRS
    core_structs.h
  DEPENDS
    libc.src.__support.CPP.string_view
    libc.src.__support.FPUtil.fp_bits
)

add_header_library(
  parser
  HDRS
    parser.h
  DEPENDS
    .core_structs
    libc.src.__support.arg_list
    libc.src.__support.ctype_utils
    libc.src.__support.str_to_integer
    libc.src.__support.CPP.algorithm
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.optional
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.type_traits
    libc.src.__support.common
)

add_object_library(
  writer
  SRCS
    writer.cpp
  HDRS
    writer.h
  DEPENDS
    .core_structs
    libc.src.__support.CPP.string_view
    libc.src.__support.macros.optimization
    libc.src.string.memory_utils.inline_memcpy
    libc.src.string.memory_utils.inline_memset
)

add_object_library(
  converter
  SRCS
    converter.cpp
  HDRS
    converter.h
    converter_atlas.h
    converter_utils.h
    string_converter.h
    char_converter.h
    int_converter.h
    ptr_converter.h
    write_int_converter.h
    float_inf_nan_converter.h
    float_hex_converter.h
    float_dec_converter.h
    fixed_converter.h #TODO: Check if this should be disabled when fixed unavail
  DEPENDS
    .core_structs
    .printf_config
    .writer
    libc.src.__support.big_int
    libc.src.__support.common
    libc.src.__support.CPP.limits
    libc.src.__support.CPP.span
    libc.src.__support.CPP.string_view
    libc.src.__support.float_to_string
    libc.src.__support.FPUtil.fenv_impl
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.rounding_mode
    libc.src.__support.integer_to_string
    libc.src.__support.libc_assert
    libc.src.__support.uint128
)


add_object_library(
  printf_main
  SRCS
    printf_main.cpp
  HDRS
    printf_main.h
  DEPENDS
    .parser
    .converter
    .writer
    .core_structs
    libc.src.__support.arg_list
)

if(NOT (TARGET libc.src.__support.File.file) AND LLVM_LIBC_FULL_BUILD)
  # Not all platforms have a file implementation. If file is unvailable, and a
  # full build is requested, then we must skip all file based printf sections.
  return()
endif()

add_header_library(
  vfprintf_internal
  HDRS
    vfprintf_internal.h
  DEPENDS
    libc.include.stdio
    libc.src.__support.File.file
    libc.src.__support.arg_list
    libc.src.stdio.printf_core.printf_main
    libc.src.stdio.printf_core.writer
  ${use_system_file}
)
