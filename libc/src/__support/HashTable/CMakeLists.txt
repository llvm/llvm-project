add_header_library(
  bitmask
  HDRS
    bitmask.h
  FLAGS
    EXPLICIT_SIMD_OPT
  DEPENDS
    libc.src.__support.common
    libc.src.__support.CPP.bit
    libc.src.__support.macros.properties.cpu_features
)

list(FIND TARGET_ENTRYPOINT_NAME_LIST getrandom getrandom_index)
if (NOT ${getrandom_index} EQUAL -1)
  message(STATUS "Using getrandom for hashtable randomness")
  set(randomness_compile_flags -DLIBC_HASHTABLE_USE_GETRANDOM)
  set(randomness_extra_depends 
    libc.src.sys.random.getrandom libc.src.errno.errno)
endif()


add_header_library(
  table
  HDRS
    table.h
  DEPENDS
    .bitmask
    libc.include.llvm-libc-types.ENTRY
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.new
    libc.src.__support.hash
    libc.src.__support.macros.attributes
    libc.src.__support.macros.optimization
    libc.src.__support.memory_size
    libc.src.string.memset
    libc.src.string.strcmp
    libc.src.string.strlen
)

add_header_library(
  randomness
  HDRS
    randomness.h
  COMPILE_OPTIONS
    ${randomness_compile_flags}
  DEPENDS
    libc.src.__support.hash
    libc.src.__support.common
    ${randomness_extra_depends}
)
