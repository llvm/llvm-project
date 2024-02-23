add_entrypoint_object(
  atoi
  SRCS
    atoi.cpp
  HDRS
    atoi.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.str_to_integer
)

add_entrypoint_object(
  atof
  SRCS
    atof.cpp
  HDRS
    atof.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.str_to_float
)

add_entrypoint_object(
  atol
  SRCS
    atol.cpp
  HDRS
    atol.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.str_to_integer
)

add_entrypoint_object(
  atoll
  SRCS
    atoll.cpp
  HDRS
    atoll.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.str_to_integer
)

add_entrypoint_object(
  getenv
  SRCS
    getenv.cpp
  HDRS
    getenv.h
  DEPENDS
    libc.config.linux.app_h
)

add_entrypoint_object(
  strtof
  SRCS
    strtof.cpp
  HDRS
    strtof.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.str_to_float
)

add_entrypoint_object(
  strtod
  SRCS
    strtod.cpp
  HDRS
    strtod.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.str_to_float
)

add_entrypoint_object(
  strtold
  SRCS
    strtold.cpp
  HDRS
    strtold.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.str_to_float
)

add_entrypoint_object(
  strtol
  SRCS
    strtol.cpp
  HDRS
    strtol.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.str_to_integer
)

add_entrypoint_object(
  strtoll
  SRCS
    strtoll.cpp
  HDRS
    strtoll.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.str_to_integer
)

add_entrypoint_object(
  strtoul
  SRCS
    strtoul.cpp
  HDRS
    strtoul.h
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.str_to_integer
)

add_entrypoint_object(
  strtoull
  SRCS
    strtoull.cpp
  HDRS
    strtoull.h
  DEPENDS
    libc.src.errno.errno
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

add_header_library(
  qsort_util
  HDRS
    qsort_util.h
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
    .qsort_util
    libc.include.stdlib
)

add_entrypoint_object(
  qsort_r
  SRCS
    qsort_r.cpp
  HDRS
    qsort_r.h
  DEPENDS
    .qsort_util
    libc.include.stdlib
)

add_object_library(
  rand_util
  SRCS
    rand_util.cpp
  HDRS
    rand_util.h
  DEPENDS
    libc.src.__support.common
)

add_entrypoint_object(
  rand
  SRCS
    rand.cpp
  HDRS
    rand.h
  DEPENDS
    .rand_util
    libc.include.stdlib
)

add_entrypoint_object(
  srand
  SRCS
    srand.cpp
  HDRS
    srand.h
  DEPENDS
    .rand_util
    libc.include.stdlib
)

if(LLVM_LIBC_INCLUDE_SCUDO)
  set(SCUDO_DEPS "")

  include(${LIBC_SOURCE_DIR}/../compiler-rt/cmake/Modules/AllSupportedArchDefs.cmake)

  # scudo distinguishes riscv32 and riscv64, so we need to translate the architecture
  set(LIBC_TARGET_ARCHITECTURE_FOR_SCUDO ${LIBC_TARGET_ARCHITECTURE})
  if(LIBC_TARGET_ARCHITECTURE_IS_RISCV64)
    set(LIBC_TARGET_ARCHITECTURE_FOR_SCUDO riscv64)
  elseif(LIBC_TARGET_ARCHITECTURE_IS_RISCV32)
    set(LIBC_TARGET_ARCHITECTURE_FOR_SCUDO riscv32)
  endif()

  if(NOT (LIBC_TARGET_ARCHITECTURE_FOR_SCUDO IN_LIST ALL_SCUDO_STANDALONE_SUPPORTED_ARCH))
    message(FATAL_ERROR "Architecture ${LIBC_TARGET_ARCHITECTURE_FOR_SCUDO} is not supported by SCUDO.
      Either disable LLVM_LIBC_INCLUDE_SCUDO or change your target architecture.")
  endif()

  list(APPEND SCUDO_DEPS RTScudoStandalone.${LIBC_TARGET_ARCHITECTURE_FOR_SCUDO}
      RTScudoStandaloneCWrappers.${LIBC_TARGET_ARCHITECTURE_FOR_SCUDO})

  list(APPEND SCUDO_DEPS
    RTGwpAsan.${LIBC_TARGET_ARCHITECTURE_FOR_SCUDO}
    RTGwpAsanBacktraceLibc.${LIBC_TARGET_ARCHITECTURE_FOR_SCUDO}
    RTGwpAsanSegvHandler.${LIBC_TARGET_ARCHITECTURE_FOR_SCUDO}
    )

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
    aligned_alloc
    DEPENDS
      ${SCUDO_DEPS}
  )
  add_entrypoint_external(
    free
    DEPENDS
      ${SCUDO_DEPS}
  )
elseif(LIBC_TARGET_OS_IS_GPU)
  add_entrypoint_external(
    calloc
  )
  add_entrypoint_external(
    realloc
  )
  add_entrypoint_external(
    aligned_alloc
  )
else()
  add_entrypoint_external(
    malloc
  )
  add_entrypoint_external(
    free
  )
  add_entrypoint_external(
    calloc
  )
  add_entrypoint_external(
    realloc
  )
  add_entrypoint_external(
    aligned_alloc
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
  SRCS
    _Exit.cpp
  HDRS
    _Exit.h
  DEPENDS
    libc.include.stdlib
    libc.src.__support.OSUtil.osutil
)

add_entrypoint_object(
  atexit
  SRCS
    atexit.cpp
  HDRS
    atexit.h
  CXX_STANDARD
    20 # For constinit of the atexit callback list.
  DEPENDS
    libc.src.__support.fixedvector
    libc.src.__support.blockstore
    libc.src.__support.threads.mutex
    libc.src.__support.CPP.new
)

add_entrypoint_object(
  exit
  SRCS
    exit.cpp
  HDRS
    exit.h
  DEPENDS
    ._Exit
    .atexit
    libc.src.__support.OSUtil.osutil
)

add_entrypoint_object(
  abort
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.abort
)

if(LIBC_TARGET_OS_IS_GPU)
  add_entrypoint_object(
    malloc
    ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.malloc
  )

  add_entrypoint_object(
    free
    ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.free
  )
endif()
