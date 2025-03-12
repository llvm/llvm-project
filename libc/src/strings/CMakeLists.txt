function(add_bcmp bcmp_name)
  add_implementation(bcmp ${bcmp_name}
    SRCS ${LIBC_SOURCE_DIR}/src/strings/bcmp.cpp
    HDRS ${LIBC_SOURCE_DIR}/src/strings/bcmp.h
    DEPENDS
      libc.src.string.memory_utils.memory_utils
    ${ARGN}
  )
endfunction()

if(${LIBC_TARGET_ARCHITECTURE_IS_X86})
  add_bcmp(bcmp_x86_64_opt_sse2   COMPILE_OPTIONS -march=k8             REQUIRE SSE2)
  add_bcmp(bcmp_x86_64_opt_sse4   COMPILE_OPTIONS -march=nehalem        REQUIRE SSE4_2)
  add_bcmp(bcmp_x86_64_opt_avx2   COMPILE_OPTIONS -march=haswell        REQUIRE AVX2)
  add_bcmp(bcmp_x86_64_opt_avx512 COMPILE_OPTIONS -march=skylake-avx512 REQUIRE AVX512BW)
  add_bcmp(bcmp_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_bcmp(bcmp)
elseif(LIBC_TARGET_OS_IS_GPU)
  add_bcmp(bcmp)
else()
  add_bcmp(bcmp_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_bcmp(bcmp)
endif()

function(add_bzero bzero_name)
  add_implementation(bzero ${bzero_name}
    SRCS ${LIBC_SOURCE_DIR}/src/strings/bzero.cpp
    HDRS ${LIBC_SOURCE_DIR}/src/strings/bzero.h
    DEPENDS
      libc.src.string.memory_utils.inline_memset
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
elseif(LIBC_TARGET_OS_IS_GPU)
  add_bzero(bzero)
else()
  add_bzero(bzero_opt_host          COMPILE_OPTIONS ${LIBC_COMPILE_OPTIONS_NATIVE})
  add_bzero(bzero)
endif()

add_entrypoint_object(
  bcopy
  SRCS
    bcopy.cpp
  HDRS
    bcopy.h
)

add_entrypoint_object(
  index
  SRCS
    index.cpp
  HDRS
    index.h
  DEPENDS
    libc.src.string.string_utils
)

add_entrypoint_object(
  rindex
  SRCS
    rindex.cpp
  HDRS
    rindex.h
  DEPENDS
    libc.src.string.string_utils
)

add_entrypoint_object(
  strcasecmp
  SRCS
    strcasecmp.cpp
  HDRS
    strcasecmp.h
  DEPENDS
    libc.src.__support.ctype_utils
    libc.src.string.memory_utils.inline_strcmp
)

add_entrypoint_object(
  strncasecmp
  SRCS
    strncasecmp.cpp
  HDRS
    strncasecmp.h
  DEPENDS
    libc.src.__support.ctype_utils
    libc.src.string.memory_utils.inline_strcmp
)
