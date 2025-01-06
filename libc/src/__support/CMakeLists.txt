add_subdirectory(CPP)
add_subdirectory(macros)

add_header_library(
  block
  HDRS
    block.h
  DEPENDS
    libc.src.__support.CPP.algorithm
    libc.src.__support.CPP.limits
    libc.src.__support.CPP.new
    libc.src.__support.CPP.optional
    libc.src.__support.CPP.span
    libc.src.__support.CPP.type_traits
)

add_object_library(
  freelist
  HDRS
    freelist.h
  SRCS
    freelist.cpp
  DEPENDS
    .block
    libc.src.__support.fixedvector
    libc.src.__support.CPP.array
    libc.src.__support.CPP.cstddef
    libc.src.__support.CPP.new
    libc.src.__support.CPP.span
)

add_object_library(
  freetrie
  HDRS
    freetrie.h
  SRCS
    freetrie.cpp
  DEPENDS
    .block
    .freelist
)

add_header_library(
  freestore
  HDRS
    freestore.h
  DEPENDS
    .freetrie
)

add_object_library(
  freelist_heap
  SRCS
    freelist_heap.cpp
  HDRS
    freelist_heap.h
  COMPILE_OPTIONS
    -DLIBC_FREELIST_MALLOC_SIZE=${LIBC_CONF_FREELIST_MALLOC_BUFFER_SIZE}
  DEPENDS
    .block
    .freelist
    .freestore
    .freetrie
    libc.src.__support.CPP.cstddef
    libc.src.__support.CPP.array
    libc.src.__support.CPP.optional
    libc.src.__support.CPP.span
    libc.src.__support.libc_assert
    libc.src.string.memory_utils.inline_memcpy
    libc.src.string.memory_utils.inline_memset
)

add_header_library(
  blockstore
  HDRS
    blockstore.h
  DEPENDS
    .libc_assert
    libc.src.__support.CPP.new
)

add_header_library(
  common
  HDRS
    common.h
    endian_internal.h
    macros/properties/architectures.h
    macros/attributes.h
    macros/properties/cpu_features.h
)

add_header_library(
  bit
  HDRS
    bit.h
  DEPENDS
    libc.src.__support.macros.attributes
    libc.src.__support.CPP.type_traits
)

add_header_library(
  math_extras
  HDRS
    math_extras.h
  DEPENDS
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.limits
    libc.src.__support.CPP.type_traits
    libc.src.__support.macros.attributes
    libc.src.__support.macros.config
)

add_header_library(
  sign
  HDRS
    sign.h
  DEPENDS
    libc.src.__support.macros.attributes
)

add_header_library(
  error_or
  HDRS
    error_or.h
  DEPENDS
    libc.src.__support.CPP.expected
)

add_header_library(
  c_string
  HDRS
    c_string.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.CPP.string
)

add_header_library(
  ctype_utils
  HDRS
    ctype_utils.h
)

add_header_library(
  wctype_utils
  HDRS
    wctype_utils.h
  DEPENDS
    libc.hdr.types.wint_t
)

add_header_library(
  str_to_num_result
  HDRS
    str_to_num_result.h
  DEPENDS
    libc.src.__support.macros.attributes
)

add_header_library(
  str_to_integer
  HDRS
    str_to_integer.h
  DEPENDS
    .ctype_utils
    .str_to_num_result
    libc.src.errno.errno
    libc.src.__support.CPP.limits
    libc.src.__support.CPP.type_traits
    libc.src.__support.common
)

add_header_library(
  integer_to_string
  HDRS
    integer_to_string.h
  DEPENDS
    .big_int
    libc.src.__support.common
    libc.src.__support.CPP.algorithm
    libc.src.__support.CPP.limits
    libc.src.__support.CPP.span
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.type_traits
)

add_header_library(
  float_to_string
  HDRS
    float_to_string.h
    ryu_constants.h
    ryu_long_double_constants.h
  DEPENDS
    .libc_assert
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
    .str_to_num_result
    .uint128
    libc.src.__support.common
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.limits
    libc.src.__support.CPP.optional
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.FPUtil.rounding_mode
    libc.src.__support.macros.config
    libc.src.__support.macros.null_check
    libc.src.__support.macros.optimization
    libc.src.errno.errno
)

add_header_library(
  integer_literals
  HDRS
    integer_literals.h
  DEPENDS
    .uint128
    libc.src.__support.CPP.limits
)

add_header_library(
  complex_type
  HDRS
    complex_type.h
  DEPENDS
    libc.src.__support.CPP.bit
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.macros.properties.types
    libc.src.__support.macros.properties.complex_types
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
    libc.hdr.func.free
    libc.hdr.func.malloc
    libc.hdr.func.realloc
    libc.src.__support.common
)

add_header_library(
  number_pair
  HDRS
    number_pair.h
  DEPENDS
    libc.src.__support.CPP.type_traits
)

add_header_library(
  big_int
  HDRS
    big_int.h
  DEPENDS
    .math_extras
    .number_pair
    libc.src.__support.CPP.array
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.type_traits
    libc.src.__support.macros.optimization
    libc.src.__support.macros.properties.types
)

add_header_library(
  uint128
  HDRS
    uint128.h
  DEPENDS
    .big_int
    libc.src.__support.macros.properties.types
)

add_header_library(
  libc_assert
  HDRS
    libc_assert.h
  DEPENDS
    .integer_to_string
    libc.src.__support.OSUtil.osutil
    libc.src.__support.macros.optimization
)

add_header_library(
  hash
  HDRS
    hash.h
  DEPENDS
    .uint128
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.limits
    libc.src.__support.macros.attributes
)

add_header_library(
  memory_size
  HDRS
    memory_size.h
  DEPENDS
    libc.src.__support.CPP.type_traits
    libc.src.__support.CPP.limits
    libc.src.__support.macros.optimization
    libc.src.__support.macros.attributes
    libc.src.__support.macros.config
)

add_header_library(
  intrusive_list
  HDRS
    intrusive_list.h
  DEPENDS
    libc.src.__support.macros.attributes
)

add_subdirectory(FPUtil)
add_subdirectory(OSUtil)
add_subdirectory(StringUtil)
add_subdirectory(GPU)
add_subdirectory(RPC)

# Thread support is used by other "File". So, we add the "threads"
# before "File".
add_subdirectory(threads)

add_subdirectory(File)

add_subdirectory(HashTable)

add_subdirectory(fixed_point)

add_subdirectory(time)
