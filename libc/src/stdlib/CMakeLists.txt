add_entrypoint_object(
  atoi
  SRCS
    atoi.cpp
  HDRS
    atoi.h
  DEPENDS
    libc.src.__support.str_to_integer
)

add_entrypoint_object(
  atof
  SRCS
    atof.cpp
  HDRS
    atof.h
  DEPENDS
    libc.src.__support.str_to_float
)

add_entrypoint_object(
  atol
  SRCS
    atol.cpp
  HDRS
    atol.h
  DEPENDS
    libc.src.__support.str_to_integer
)

add_entrypoint_object(
  atoll
  SRCS
    atoll.cpp
  HDRS
    atoll.h
  DEPENDS
    libc.src.__support.str_to_integer
)

add_entrypoint_object(
  strtof
  SRCS
    strtof.cpp
  HDRS
    strtof.h
  DEPENDS
    libc.src.__support.str_to_float
)

add_entrypoint_object(
  strtod
  SRCS
    strtod.cpp
  HDRS
    strtod.h
  DEPENDS
    libc.src.__support.str_to_float
)

add_entrypoint_object(
  strtold
  SRCS
    strtold.cpp
  HDRS
    strtold.h
  DEPENDS
    libc.src.__support.str_to_float
)

add_entrypoint_object(
  strtol
  SRCS
    strtol.cpp
  HDRS
    strtol.h
  DEPENDS
    libc.src.__support.str_to_integer
)

add_entrypoint_object(
  strtoll
  SRCS
    strtoll.cpp
  HDRS
    strtoll.h
  DEPENDS
    libc.src.__support.str_to_integer
)

add_entrypoint_object(
  strtoul
  SRCS
    strtoul.cpp
  HDRS
    strtoul.h
  DEPENDS
    libc.src.__support.str_to_integer
)

add_entrypoint_object(
  strtoull
  SRCS
    strtoull.cpp
  HDRS
    strtoull.h
  DEPENDS
    libc.src.__support.str_to_integer
)

add_entrypoint_object(
  abs
  SRCS
    abs.cpp
  HDRS
    abs.h
  DEPENDS
    libc.src.__support.integer_operations
)

add_entrypoint_object(
  labs
  SRCS
    labs.cpp
  HDRS
    labs.h
  DEPENDS
    libc.src.__support.integer_operations
)

add_entrypoint_object(
  llabs
  SRCS
    llabs.cpp
  HDRS
    llabs.h
  DEPENDS
    libc.src.__support.integer_operations
)

add_entrypoint_object(
  div
  SRCS
    div.cpp
  HDRS
    div.h
  DEPENDS
    libc.include.stdlib
    libc.src.__support.integer_operations
)

add_entrypoint_object(
  ldiv
  SRCS
    ldiv.cpp
  HDRS
    ldiv.h
  DEPENDS
    libc.include.stdlib
    libc.src.__support.integer_operations
)

add_entrypoint_object(
  lldiv
  SRCS
    lldiv.cpp
  HDRS
    lldiv.h
  DEPENDS
    libc.include.stdlib
    libc.src.__support.integer_operations
)

add_entrypoint_object(
  bsearch
  SRCS
    bsearch.cpp
  HDRS
    bsearch.h
  DEPENDS
    libc.include.stdlib
)

add_entrypoint_object(
  qsort
  SRCS
    qsort.cpp
  HDRS
    qsort.h
  DEPENDS
    libc.include.stdlib
)

if(LLVM_LIBC_INCLUDE_SCUDO)
  set(SCUDO_DEPS "")

  include(${LIBC_SOURCE_DIR}/../compiler-rt/cmake/Modules/AllSupportedArchDefs.cmake)
  if(NOT (LIBC_TARGET_ARCHITECTURE IN_LIST ALL_SCUDO_STANDALONE_SUPPORTED_ARCH))
    message(FATAL_ERROR "Architecture ${LIBC_TARGET_ARCHITECTURE} is not supported by SCUDO. 
      Either disable LLVM_LIBC_INCLUDE_SCUDO or change your target architecture.")
  endif()
  list(APPEND SCUDO_DEPS RTScudoStandalone.${LIBC_TARGET_ARCHITECTURE} 
       RTScudoStandaloneCWrappers.${LIBC_TARGET_ARCHITECTURE})
  if((LIBC_TARGET_ARCHITECTURE IN_LIST ALL_GWP_ASAN_SUPPORTED_ARCH) 
      AND COMPILER_RT_BUILD_GWP_ASAN)
    list(APPEND SCUDO_DEPS RTGwpAsan.${LIBC_TARGET_ARCHITECTURE} 
                            RTGwpAsanBacktraceLibc.${LIBC_TARGET_ARCHITECTURE} 
                            RTGwpAsanSegvHandler.${LIBC_TARGET_ARCHITECTURE})
  elseif(COMPILER_RT_BUILD_GWP_ASAN)
    message(WARNING "Architecture ${LIBC_TARGET_ARCHITECTURE} is not supported by GWP-ASan. Skipping.")
  endif()

  add_entrypoint_external(
    malloc
    DEPENDS
      ${SCUDO_DEPS}
  )
  add_entrypoint_external(
    calloc
    DEPENDS
      ${SCUDO_DEPS}
  )
  add_entrypoint_external(
    realloc
    DEPENDS
      ${SCUDO_DEPS}
  )
  add_entrypoint_external(
    free
    DEPENDS
      ${SCUDO_DEPS}
  )

endif()

if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  _Exit
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}._Exit
)

# add_entrypoint_object(
#   abort
#   SRCS
#     abort.cpp
#   HDRS
#     abort.h
#   DEPENDS
#     libc.include.stdlib
#     libc.src.signal.raise
#     ._Exit
# )
