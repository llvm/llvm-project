add_subdirectory(memory_utils)

add_entrypoint_object(
  strcat
  SRCS
    strcat.cpp
  HDRS
    strcat.h
  DEPENDS
    strcpy
    string_h
)

add_entrypoint_object(
  strcpy
  SRCS
    strcpy.cpp
  HDRS
    strcpy.h
  DEPENDS
    string_h
)

# ------------------------------------------------------------------------------
# memcpy
# ------------------------------------------------------------------------------

# include the relevant architecture specific implementations
if(${LIBC_TARGET_MACHINE} STREQUAL "x86_64")
  set(LIBC_MEMCPY_IMPL_FOLDER "x86")
else()
  set(LIBC_MEMCPY_IMPL_FOLDER ${LIBC_TARGET_MACHINE})
endif()

add_gen_header(
  memcpy_arch_specific
  DEF_FILE
    memcpy_arch_specific.h.def
  GEN_HDR
    memcpy_arch_specific.h
  PARAMS
    memcpy_arch_specific=${LIBC_MEMCPY_IMPL_FOLDER}/memcpy_arch_specific.h.inc
  DATA_FILES
    ${LIBC_MEMCPY_IMPL_FOLDER}/memcpy_arch_specific.h.inc
)

# Helper to define an implementation of memcpy.
# - Computes flags to satisfy required/rejected features and arch,
# - Declares an entry point,
# - Attach the REQUIRE_CPU_FEATURES property to the target,
# - Add the target to `memcpy_implementations` global property for tests.
function(add_memcpy memcpy_name)
  cmake_parse_arguments(
    "ADD_MEMCPY"
    "" # Optional arguments
    "MARCH" # Single value arguments
    "REQUIRE;REJECT" # Multi value arguments
    ${ARGN})
  compute_flags(flags
    MARCH ${ADD_MEMCPY_MARCH}
    REQUIRE ${ADD_MEMCPY_REQUIRE}
    REJECT ${ADD_MEMCPY_REJECT}
  )
  add_entrypoint_object(
    ${memcpy_name}
    SRCS ${LIBC_SOURCE_DIR}/src/string/memcpy.cpp
    HDRS ${LIBC_SOURCE_DIR}/src/string/memcpy.h
    DEPENDS
      string_h
      memory_utils
      memcpy_arch_specific
    COMPILE_OPTIONS
      -fno-builtin-memcpy
      ${flags}
  )
  set_target_properties(${memcpy_name} PROPERTIES REQUIRE_CPU_FEATURES "${ADD_MEMCPY_REQUIRE}")
  get_property(all GLOBAL PROPERTY memcpy_implementations)
  list(APPEND all ${memcpy_name})
  set_property(GLOBAL PROPERTY memcpy_implementations "${all}")
endfunction()

add_subdirectory(${LIBC_MEMCPY_IMPL_FOLDER})
add_memcpy(memcpy MARCH native)
