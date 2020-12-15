add_subdirectory(memory_utils)

add_header_library(
  string_utils
  HDRS
    string_utils.h
  DEPENDS
    libc.utils.CPP.standalone_cpp
)

add_entrypoint_object(
  strcat
  SRCS
    strcat.cpp
  HDRS
    strcat.h
  DEPENDS
    .strcpy
    .string_utils
)

add_entrypoint_object(
  strcpy
  SRCS
    strcpy.cpp
  HDRS
    strcpy.h
  DEPENDS
    .memcpy
    .string_utils
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

add_entrypoint_object(
  strcmp
  SRCS
    strcmp.cpp
  HDRS
    strcmp.h
)

add_entrypoint_object(
  memchr
  SRCS
    memchr.cpp
  HDRS
    memchr.h
  DEPENDS
    .string_utils
)

add_entrypoint_object(
  memcmp
  SRCS
    memcmp.cpp
  HDRS
    memcmp.h
)

add_entrypoint_object(
  strchr
  SRCS
    strchr.cpp
  HDRS
    strchr.h
)

add_entrypoint_object(
  strstr
  SRCS
    strstr.cpp
  HDRS
    strstr.h
)

add_entrypoint_object(
  strncpy
  SRCS
    strncpy.cpp
  HDRS
    strncpy.h
)

add_entrypoint_object(
  strnlen
  SRCS
    strnlen.cpp
  HDRS
    strnlen.h
  DEPENDS
    .string_utils
)

add_entrypoint_object(
  memrchr
  SRCS
    memrchr.cpp
  HDRS
    memrchr.h
)

add_entrypoint_object(
  strrchr
  SRCS
    strrchr.cpp
  HDRS
    strrchr.h
)

add_entrypoint_object(
  strcspn
  SRCS
    strcspn.cpp
  HDRS
    strcspn.h
  DEPENDS
    .string_utils
)

add_entrypoint_object(
  strspn
  SRCS
    strspn.cpp
  HDRS
    strspn.h
  DEPENDS
    libc.utils.CPP.standalone_cpp
)

add_entrypoint_object(
  strpbrk
  SRCS
    strpbrk.cpp
  HDRS
    strpbrk.h
  DEPENDS
    .string_utils
)

add_entrypoint_object(
  strtok
  SRCS
    strtok.cpp
  HDRS
    strtok.h
  DEPENDS
    .string_utils
)

add_entrypoint_object(
  strtok_r
  SRCS
    strtok_r.cpp
  HDRS
    strtok_r.h
  DEPENDS
    .string_utils
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
    COMPILE_OPTIONS ${ADD_IMPL_COMPILE_OPTIONS} ${flags} -O2
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
  set(MEMCPY_SRC ${LIBC_SOURCE_DIR}/src/string/x86/memcpy.cpp)
else()
  set(LIBC_STRING_TARGET_ARCH ${LIBC_TARGET_MACHINE})
  set(MEMCPY_SRC ${LIBC_SOURCE_DIR}/src/string/memcpy.cpp)
endif()

function(add_memcpy memcpy_name)
  add_implementation(memcpy ${memcpy_name}
    SRCS ${MEMCPY_SRC}
    HDRS ${LIBC_SOURCE_DIR}/src/string/memcpy.h
    DEPENDS
      .memory_utils.memory_utils
      libc.include.string
    COMPILE_OPTIONS
      -fno-builtin-memcpy
    ${ARGN}
  )
endfunction()

if(${LIBC_STRING_TARGET_ARCH} STREQUAL "x86")
  add_memcpy(memcpy MARCH native)
else()
  add_memcpy(memcpy)
endif()

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

if(${LIBC_STRING_TARGET_ARCH} STREQUAL "x86")
  add_memset(memset MARCH native)
else()
  add_memset(memset)
endif()

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

if(${LIBC_STRING_TARGET_ARCH} STREQUAL "x86")
  add_bzero(bzero MARCH native)
else()
  add_bzero(bzero)
endif()

# ------------------------------------------------------------------------------
# Add all other relevant implementations for the native target.
# ------------------------------------------------------------------------------

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_STRING_TARGET_ARCH})
  include(${LIBC_STRING_TARGET_ARCH}/CMakeLists.txt)
endif()
