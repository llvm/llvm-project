add_subdirectory(CPP)

add_header_library(
  blockstore
  HDRS
    blockstore.h
  DEPENDS
    libc.src.__support.CPP.new
)

add_header_library(
  named_pair
  HDRS
    named_pair.h
)

add_header_library(
  compiler_features
  HDRS
    macros/properties/compiler.h
)

add_header_library(
  sanitizer
  HDRS
    macros/sanitizer.h
  DEPENDS
    libc.src.__support.compiler_features
)

add_header_library(
  common
  HDRS
    common.h
    endian.h
    macros/properties/architectures.h
    macros/attributes.h
    macros/cpu_features.h
)

add_header_library(
  builtin_wrappers
  HDRS
    builtin_wrappers.h
  DEPENDS
    .named_pair
    libc.src.__support.common
    libc.src.__support.compiler_features
    libc.src.__support.CPP.type_traits
)

add_header_library(
  error_or
  HDRS
    error_or.h
  DEPENDS
    libc.src.__support.CPP.expected
)

add_header_library(
  ctype_utils
  HDRS
    ctype_utils.h
)

add_header_library(
  str_to_num_result
  HDRS
    str_to_num_result.h
)

add_header_library(
  str_to_integer
  HDRS
    str_to_integer.h
  DEPENDS
    .ctype_utils
    .str_to_num_result
    libc.include.errno
    libc.src.errno.errno
    libc.src.__support.CPP.limits
    libc.src.__support.common
)

add_header_library(
  integer_to_string
  HDRS
    integer_to_string.h
  DEPENDS
    libc.src.__support.CPP.span
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.type_traits
    libc.src.__support.common
)


add_header_library(
  float_to_string
  HDRS
    float_to_string.h
    ryu_constants.h
  DEPENDS
    libc.src.__support.CPP.type_traits
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.common
)

add_header_library(
  high_precision_decimal
  HDRS
    high_precision_decimal.h
  DEPENDS
    .str_to_integer
)

add_header_library(
  str_to_float
  HDRS
    str_to_float.h
    detailed_powers_of_ten.h
  DEPENDS
    .ctype_utils
    .high_precision_decimal
    .str_to_integer
    .uint128
    libc.include.errno
    libc.src.__support.CPP.limits
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.builtin_wrappers
    libc.src.__support.common
    libc.src.errno.errno
)


add_header_library(
  integer_operations
  HDRS
    integer_operations.h
)

add_header_library(
  arg_list
  HDRS
    arg_list.h
  DEPENDS
    libc.src.__support.common
)

add_header_library(
  fixedvector
  HDRS
    fixedvector.h
  DEPENDS
    libc.src.__support.CPP.array
)

add_header_library(
  char_vector
  HDRS
    char_vector.h
  DEPENDS
    libc.src.__support.common
)

add_header_library(
  number_pair
  HDRS
    number_pair.h
  DEPENDS
    .named_pair
    libc.src.__support.CPP.type_traits
)

add_header_library(
  integer_utils
  HDRS
    integer_utils.h
  DEPENDS
    .builtin_wrappers
    .number_pair
    libc.src.__support.CPP.type_traits
    libc.src.__support.common
)

add_header_library(
  uint
  HDRS
    UInt.h
  DEPENDS
    .builtin_wrappers
    .number_pair
    .integer_utils
    libc.src.__support.CPP.array
    libc.src.__support.CPP.type_traits
)

add_header_library(
  uint128
  HDRS
    UInt128.h
  DEPENDS
    .uint
)

add_subdirectory(FPUtil)
add_subdirectory(OSUtil)
add_subdirectory(StringUtil)

# Thread support is used by other "File". So, we add the "threads"
# before "File".
add_subdirectory(threads)

add_subdirectory(File)
