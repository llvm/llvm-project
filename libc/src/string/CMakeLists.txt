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
  strncmp
  SRCS
    strncmp.cpp
  HDRS
    strncmp.h
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
  memmove
  SRCS
    memmove.cpp
  HDRS
    memmove.h
  DEPENDS
    libc.src.__support.integer_operations
    libc.src.string.memcpy
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
    "" # Single value arguments
    "REQUIRE;SRCS;HDRS;DEPENDS;COMPILE_OPTIONS" # Multi value arguments
    ${ARGN})
  add_entrypoint_object(${impl_name}
    NAME ${name}
    SRCS ${ADD_IMPL_SRCS}
    HDRS ${ADD_IMPL_HDRS}
    DEPENDS ${ADD_IMPL_DEPENDS}
    COMPILE_OPTIONS ${ADD_IMPL_COMPILE_OPTIONS} "SHELL:-mllvm -combiner-global-alias-analysis"
  )
  get_fq_target_name(${impl_name} fq_target_name)
  set_target_properties(${fq_target_name} PROPERTIES REQUIRE_CPU_FEATURES "${ADD_IMPL_REQUIRE}")
  set_property(GLOBAL APPEND PROPERTY "${name}_implementations" "${fq_target_name}")
endfunction()

# ------------------------------------------------------------------------------
# memcpy
# ------------------------------------------------------------------------------

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

if(${LIBC_TARGET_ARCHITECTURE_IS_X86})
  set(MEMCPY_SRC ${LIBC_SOURCE_DIR}/src/string/x86_64/memcpy.cpp)
  add_memcpy(memcpy_x86_64_opt_sse2   COMPILE_OPTIONS -march=k8             REQUIRE SSE2)
  add_memcpy(memcpy_x86_64_opt_sse4   COMPILE_OPTIONS -march=nehalem        REQUIRE SSE4_2)
  add_memcpy(memcpy_x86_64_opt_avx2   COMPILE_OPTIONS -march=haswell        REQUIRE AVX2)
  add_memcpy(memcpy_x86_64_opt_avx512 COMPILE_OPTIONS -march=skylake-avx512 REQUIRE AVX512F)
  add_memcpy(memcpy_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_memcpy(memcpy)
elseif(${LIBC_TARGET_ARCHITECTURE_IS_AARCH64})
  set(MEMCPY_SRC ${LIBC_SOURCE_DIR}/src/string/aarch64/memcpy.cpp)
  # Disable tail merging as it leads to lower performance.
  # Note that '-mllvm' needs to be prefixed with 'SHELL:' to prevent CMake flag deduplication.
  add_memcpy(memcpy_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE}
                                      COMPILE_OPTIONS "SHELL:-mllvm --tail-merge-threshold=0")
  add_memcpy(memcpy                   COMPILE_OPTIONS "SHELL:-mllvm --tail-merge-threshold=0")
else()
  set(MEMCPY_SRC ${LIBC_SOURCE_DIR}/src/string/memcpy.cpp)
  add_memcpy(memcpy_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
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

if(${LIBC_TARGET_ARCHITECTURE_IS_X86})
  add_memset(memset_x86_64_opt_sse2   COMPILE_OPTIONS -march=k8             REQUIRE SSE2)
  add_memset(memset_x86_64_opt_sse4   COMPILE_OPTIONS -march=nehalem        REQUIRE SSE4_2)
  add_memset(memset_x86_64_opt_avx2   COMPILE_OPTIONS -march=haswell        REQUIRE AVX2)
  add_memset(memset_x86_64_opt_avx512 COMPILE_OPTIONS -march=skylake-avx512 REQUIRE AVX512F)
  add_memset(memset_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_memset(memset)
else()
  add_memset(memset_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
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
      -fno-builtin-bzero
    ${ARGN}
  )
endfunction()

if(${LIBC_TARGET_ARCHITECTURE_IS_X86})
  add_bzero(bzero_x86_64_opt_sse2   COMPILE_OPTIONS -march=k8             REQUIRE SSE2)
  add_bzero(bzero_x86_64_opt_sse4   COMPILE_OPTIONS -march=nehalem        REQUIRE SSE4_2)
  add_bzero(bzero_x86_64_opt_avx2   COMPILE_OPTIONS -march=haswell        REQUIRE AVX2)
  add_bzero(bzero_x86_64_opt_avx512 COMPILE_OPTIONS -march=skylake-avx512 REQUIRE AVX512F)
  add_bzero(bzero_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_bzero(bzero)
else()
  add_bzero(bzero_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_bzero(bzero)
endif()

# ------------------------------------------------------------------------------
# memcmp
# ------------------------------------------------------------------------------

function(add_memcmp memcmp_name)
  add_implementation(memcmp ${memcmp_name}
    SRCS ${LIBC_MEMCMP_SRC}
    HDRS ${LIBC_SOURCE_DIR}/src/string/memcmp.h
    DEPENDS
      .memory_utils.memory_utils
      libc.include.string
    COMPILE_OPTIONS
      -fno-builtin-memcmp
    ${ARGN}
  )
endfunction()

if(${LIBC_TARGET_ARCHITECTURE_IS_X86})
  set(LIBC_MEMCMP_SRC ${LIBC_SOURCE_DIR}/src/string/memcmp.cpp)
  add_memcmp(memcmp_x86_64_opt_sse2   COMPILE_OPTIONS -march=k8             REQUIRE SSE2)
  add_memcmp(memcmp_x86_64_opt_sse4   COMPILE_OPTIONS -march=nehalem        REQUIRE SSE4_2)
  add_memcmp(memcmp_x86_64_opt_avx2   COMPILE_OPTIONS -march=haswell        REQUIRE AVX2)
  add_memcmp(memcmp_x86_64_opt_avx512 COMPILE_OPTIONS -march=skylake-avx512 REQUIRE AVX512F)
  add_memcmp(memcmp_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_memcmp(memcmp)
elseif(${LIBC_TARGET_ARCHITECTURE_IS_AARCH64})
  set(LIBC_MEMCMP_SRC ${LIBC_SOURCE_DIR}/src/string/aarch64/memcmp.cpp)
  add_memcmp(memcmp)
  add_memcmp(memcmp_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
else()
  set(LIBC_MEMCMP_SRC ${LIBC_SOURCE_DIR}/src/string/memcmp.cpp)
  add_memcmp(memcmp_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_memcmp(memcmp)
endif()
