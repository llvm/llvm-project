add_subdirectory(memory_utils)

add_entrypoint_object(
  strcat
  SRCS
    strcat.cpp
  HDRS
    strcat.h
  DEPENDS
    .strcpy
    .strlen
    libc.include.string
)

add_entrypoint_object(
  strcpy
  SRCS
    strcpy.cpp
  HDRS
    strcpy.h
  DEPENDS
    .memcpy
    .strlen
    libc.include.string
)

add_entrypoint_object(
  strlen
  SRCS
    strlen.cpp
  HDRS
    strlen.h
  DEPENDS
    libc.include.string
)

# Helper to define a function with multiple implementations
# - Computes flags to satisfy required/rejected features and arch,
# - Declares an entry point,
# - Attach the REQUIRE_CPU_FEATURES property to the target,
# - Add the fully qualified target to `${name}_implementations` global property for tests.
function(add_implementation name impl_name)
  cmake_parse_arguments(
    "ADD_IMPL"
    "" # Optional arguments
    "MARCH" # Single value arguments
    "REQUIRE;REJECT;SRCS;HDRS;DEPENDS;COMPILE_OPTIONS" # Multi value arguments
    ${ARGN})
  compute_flags(flags
    MARCH ${ADD_IMPL_MARCH}
    REQUIRE ${ADD_IMPL_REQUIRE}
    REJECT ${ADD_IMPL_REJECT}
  )
  add_entrypoint_object(${impl_name}
    NAME ${name}
    SRCS ${ADD_IMPL_SRCS}
    HDRS ${ADD_IMPL_HDRS}
    DEPENDS ${ADD_IMPL_DEPENDS}
    COMPILE_OPTIONS ${ADD_IMPL_COMPILE_OPTIONS} ${flags}
  )
  get_fq_target_name(${impl_name} fq_target_name)
  set_target_properties(${fq_target_name} PROPERTIES REQUIRE_CPU_FEATURES "${ADD_IMPL_REQUIRE}")
  set_property(GLOBAL APPEND PROPERTY "${name}_implementations" "${fq_target_name}")
endfunction()

# ------------------------------------------------------------------------------
# memcpy
# ------------------------------------------------------------------------------

# include the relevant architecture specific implementations
if(${LIBC_TARGET_MACHINE} STREQUAL "x86_64")
  set(LIBC_STRING_TARGET_ARCH "x86")
else()
  set(LIBC_STRING_TARGET_ARCH ${LIBC_TARGET_MACHINE})
endif()

function(add_memcpy memcpy_name)
  add_implementation(memcpy ${memcpy_name}
    SRCS ${LIBC_SOURCE_DIR}/src/string/${LIBC_STRING_TARGET_ARCH}/memcpy.cpp
    HDRS ${LIBC_SOURCE_DIR}/src/string/memcpy.h
    DEPENDS
      .memory_utils.memory_utils
      libc.include.string
    COMPILE_OPTIONS
      -fno-builtin-memcpy
    ${ARGN}
  )
endfunction()

add_memcpy(memcpy MARCH native)

# ------------------------------------------------------------------------------
# memset
# ------------------------------------------------------------------------------

function(add_memset memset_name)
  add_implementation(memset ${memset_name}
    SRCS ${LIBC_SOURCE_DIR}/src/string/memset.cpp
    HDRS ${LIBC_SOURCE_DIR}/src/string/memset.h
    DEPENDS
      .memory_utils.memory_utils
      libc.include.string
    COMPILE_OPTIONS
      -fno-builtin-memset
    ${ARGN}
  )
endfunction()

add_memset(memset MARCH native)

# ------------------------------------------------------------------------------
# bzero
# ------------------------------------------------------------------------------

function(add_bzero bzero_name)
  add_implementation(bzero ${bzero_name}
    SRCS ${LIBC_SOURCE_DIR}/src/string/bzero.cpp
    HDRS ${LIBC_SOURCE_DIR}/src/string/bzero.h
    DEPENDS
      .memory_utils.memory_utils
      libc.include.string
    COMPILE_OPTIONS
      -fno-builtin-memset
      -fno-builtin-bzero
    ${ARGN}
  )
endfunction()

add_bzero(bzero MARCH native)

# ------------------------------------------------------------------------------
# Add all other relevant implementations for the native target.
# ------------------------------------------------------------------------------

if(EXISTS ${LIBC_STRING_TARGET_ARCH})
  include(${LIBC_STRING_TARGET_ARCH}/CMakeLists.txt)
endif()
