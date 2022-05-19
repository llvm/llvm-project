add_subdirectory(memory_utils)

add_header_library(
  string_utils
  HDRS
    string_utils.h
  DEPENDS
    libc.src.__support.CPP.bitset
)

add_entrypoint_object(
  memccpy
  SRCS
    memccpy.cpp
  HDRS
    memccpy.h
)


add_entrypoint_object(
  mempcpy
  SRCS
    mempcpy.cpp
  HDRS
    mempcpy.h
  DEPENDS
    .memory_utils.memcpy_implementation
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
  memrchr
  SRCS
    memrchr.cpp
  HDRS
    memrchr.h
)

add_entrypoint_object(
  stpcpy
  SRCS
    stpcpy.cpp
  HDRS
    stpcpy.h
  DEPENDS
    .mempcpy
    .string_utils
)

add_entrypoint_object(
  stpncpy
  SRCS
    stpncpy.cpp
  HDRS
    stpncpy.h
  DEPENDS
    .memory_utils.memset_implementation
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
  strchr
  SRCS
    strchr.cpp
  HDRS
    strchr.h
)

add_entrypoint_object(
  strcmp
  SRCS
    strcmp.cpp
  HDRS
    strcmp.h
)

add_entrypoint_object(
  strcpy
  SRCS
    strcpy.cpp
  HDRS
    strcpy.h
  DEPENDS
    .memory_utils.memcpy_implementation
    .string_utils
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
  strdup
  SRCS
    strdup.cpp
  HDRS
    strdup.h
  DEPENDS
    .memory_utils.memcpy_implementation
    .string_utils
    libc.include.stdlib
)

add_entrypoint_object(
  strlcpy
  SRCS
    strlcpy.cpp
  HDRS
    strlcpy.h
  DEPENDS
    .memory_utils.memcpy_implementation
    .memory_utils.memset_implementation
    .string_utils
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

add_entrypoint_object(
  strncat
  SRCS
    strncat.cpp
  HDRS
    strncat.h
  DEPENDS
    .strncpy
    .string_utils
)

add_entrypoint_object(
  strncmp
  SRCS
    strncmp.cpp
  HDRS
    strncmp.h
)

add_entrypoint_object(
  strncpy
  SRCS
    strncpy.cpp
  HDRS
    strncpy.h
)

add_entrypoint_object(
  strndup
  SRCS
    strndup.cpp
  HDRS
    strndup.h
  DEPENDS
    .memory_utils.memcpy_implementation
    .string_utils
    libc.include.stdlib
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
  strpbrk
  SRCS
    strpbrk.cpp
  HDRS
    strpbrk.h
  DEPENDS
    .string_utils
)

add_entrypoint_object(
  strrchr
  SRCS
    strrchr.cpp
  HDRS
    strrchr.h
)

add_entrypoint_object(
  strspn
  SRCS
    strspn.cpp
  HDRS
    strspn.h
  DEPENDS
    libc.src.__support.CPP.bitset
)

add_entrypoint_object(
  strstr
  SRCS
    strstr.cpp
  HDRS
    strstr.h
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
    "REQUIRE;SRCS;HDRS;DEPENDS;COMPILE_OPTIONS;MLLVM_COMPILE_OPTIONS" # Multi value arguments
    ${ARGN})

  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    # Note that '-mllvm' needs to be prefixed with 'SHELL:' to prevent CMake flag deduplication.
    foreach(opt IN LISTS ADD_IMPL_MLLVM_COMPILE_OPTIONS)
      list(APPEND ADD_IMPL_COMPILE_OPTIONS "SHELL:-mllvm ${opt}")
    endforeach()
  endif()

  add_entrypoint_object(${impl_name}
    NAME ${name}
    SRCS ${ADD_IMPL_SRCS}
    HDRS ${ADD_IMPL_HDRS}
    DEPENDS ${ADD_IMPL_DEPENDS}
    COMPILE_OPTIONS ${ADD_IMPL_COMPILE_OPTIONS}
  )
  get_fq_target_name(${impl_name} fq_target_name)
  set_target_properties(${fq_target_name} PROPERTIES REQUIRE_CPU_FEATURES "${ADD_IMPL_REQUIRE}")
  set_property(GLOBAL APPEND PROPERTY "${name}_implementations" "${fq_target_name}")
endfunction()

# ------------------------------------------------------------------------------
# bcmp
# ------------------------------------------------------------------------------

function(add_bcmp bcmp_name)
  add_implementation(bcmp ${bcmp_name}
    SRCS ${LIBC_SOURCE_DIR}/src/string/bcmp.cpp
    HDRS ${LIBC_SOURCE_DIR}/src/string/bcmp.h
    DEPENDS
      .memory_utils.memory_utils
      libc.include.string
    ${ARGN}
  )
endfunction()

if(${LIBC_TARGET_ARCHITECTURE_IS_X86})
  add_bcmp(bcmp_x86_64_opt_sse2   COMPILE_OPTIONS -march=k8             REQUIRE SSE2)
  add_bcmp(bcmp_x86_64_opt_sse4   COMPILE_OPTIONS -march=nehalem        REQUIRE SSE4_2)
  add_bcmp(bcmp_x86_64_opt_avx2   COMPILE_OPTIONS -march=haswell        REQUIRE AVX2)
  add_bcmp(bcmp_x86_64_opt_avx512 COMPILE_OPTIONS -march=skylake-avx512 REQUIRE AVX512F)
  add_bcmp(bcmp_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_bcmp(bcmp)
else()
  add_bcmp(bcmp_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_bcmp(bcmp)
endif()

# ------------------------------------------------------------------------------
# bzero
# ------------------------------------------------------------------------------

function(add_bzero bzero_name)
  add_implementation(bzero ${bzero_name}
    SRCS ${LIBC_SOURCE_DIR}/src/string/bzero.cpp
    HDRS ${LIBC_SOURCE_DIR}/src/string/bzero.h
    DEPENDS
      .memory_utils.memset_implementation
      libc.include.string
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
    SRCS ${LIBC_SOURCE_DIR}/src/string/memcmp.cpp
    HDRS ${LIBC_SOURCE_DIR}/src/string/memcmp.h
    DEPENDS
      .memory_utils.memcmp_implementation
      libc.include.string
    ${ARGN}
  )
endfunction()

if(${LIBC_TARGET_ARCHITECTURE_IS_X86})
  add_memcmp(memcmp_x86_64_opt_sse2   COMPILE_OPTIONS -march=k8             REQUIRE SSE2)
  add_memcmp(memcmp_x86_64_opt_sse4   COMPILE_OPTIONS -march=nehalem        REQUIRE SSE4_2)
  add_memcmp(memcmp_x86_64_opt_avx2   COMPILE_OPTIONS -march=haswell        REQUIRE AVX2)
  add_memcmp(memcmp_x86_64_opt_avx512 COMPILE_OPTIONS -march=skylake-avx512 REQUIRE AVX512F)
  add_memcmp(memcmp_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_memcmp(memcmp)
elseif(${LIBC_TARGET_ARCHITECTURE_IS_AARCH64})
  add_memcmp(memcmp_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_memcmp(memcmp)
else()
  add_memcmp(memcmp_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_memcmp(memcmp)
endif()

# ------------------------------------------------------------------------------
# memcpy
# ------------------------------------------------------------------------------

function(add_memcpy memcpy_name)
  add_implementation(memcpy ${memcpy_name}
    SRCS ${LIBC_SOURCE_DIR}/src/string/memcpy.cpp
    HDRS ${LIBC_SOURCE_DIR}/src/string/memcpy.h
    DEPENDS
      .memory_utils.memcpy_implementation
      libc.include.string
    ${ARGN}
  )
endfunction()

if(${LIBC_TARGET_ARCHITECTURE_IS_X86})
  add_memcpy(memcpy_x86_64_opt_sse2   COMPILE_OPTIONS -march=k8             REQUIRE SSE2)
  add_memcpy(memcpy_x86_64_opt_sse4   COMPILE_OPTIONS -march=nehalem        REQUIRE SSE4_2)
  add_memcpy(memcpy_x86_64_opt_avx2   COMPILE_OPTIONS -march=haswell        REQUIRE AVX2)
  add_memcpy(memcpy_x86_64_opt_avx512 COMPILE_OPTIONS -march=skylake-avx512 REQUIRE AVX512F)
  add_memcpy(memcpy_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_memcpy(memcpy)
elseif(${LIBC_TARGET_ARCHITECTURE_IS_AARCH64})
  # Disable tail merging as it leads to lower performance.
  add_memcpy(memcpy_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE}
                                      MLLVM_COMPILE_OPTIONS "-tail-merge-threshold=0")
  add_memcpy(memcpy                   MLLVM_COMPILE_OPTIONS "-tail-merge-threshold=0")
else()
  add_memcpy(memcpy_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_memcpy(memcpy)
endif()

# ------------------------------------------------------------------------------
# memmove
# ------------------------------------------------------------------------------

function(add_memmove memmove_name)
  add_implementation(memmove ${memmove_name}
    SRCS ${LIBC_SOURCE_DIR}/src/string/memmove.cpp
    HDRS ${LIBC_SOURCE_DIR}/src/string/memmove.h
    DEPENDS
      .memory_utils.memory_utils
      libc.include.string
    ${ARGN}
  )
endfunction()

if(${LIBC_TARGET_ARCHITECTURE_IS_X86})
  add_memmove(memmove_x86_64_opt_sse2   COMPILE_OPTIONS -march=k8             REQUIRE SSE2)
  add_memmove(memmove_x86_64_opt_sse4   COMPILE_OPTIONS -march=nehalem        REQUIRE SSE4_2)
  add_memmove(memmove_x86_64_opt_avx2   COMPILE_OPTIONS -march=haswell        REQUIRE AVX2)
  add_memmove(memmove_x86_64_opt_avx512 COMPILE_OPTIONS -march=skylake-avx512 REQUIRE AVX512F)
  add_memmove(memmove_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_memmove(memmove)
elseif(${LIBC_TARGET_ARCHITECTURE_IS_AARCH64})
  # Disable tail merging as it leads to lower performance.
  add_memmove(memmove_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE}
                                        MLLVM_COMPILE_OPTIONS "-tail-merge-threshold=0")
  add_memmove(memmove                   MLLVM_COMPILE_OPTIONS "-tail-merge-threshold=0")
else()
  add_memmove(memmove_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_memmove(memmove)
endif()

# ------------------------------------------------------------------------------
# memset
# ------------------------------------------------------------------------------

function(add_memset memset_name)
  add_implementation(memset ${memset_name}
    SRCS ${LIBC_SOURCE_DIR}/src/string/memset.cpp
    HDRS ${LIBC_SOURCE_DIR}/src/string/memset.h
    DEPENDS
      .memory_utils.memset_implementation
      libc.include.string
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
elseif(${LIBC_TARGET_ARCHITECTURE_IS_AARCH64})
  # Disable tail merging as it leads to lower performance.
  add_memset(memset_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE}
                                      MLLVM_COMPILE_OPTIONS "-tail-merge-threshold=0")
  add_memset(memset                   MLLVM_COMPILE_OPTIONS "-tail-merge-threshold=0")
else()
  add_memset(memset_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_memset(memset)
endif()
