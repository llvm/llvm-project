add_object_library(
  base
  SRCS
    base.cpp
  HDRS
    base.h
  DEPENDS
    libc.hdr.types.size_t
    libc.src.__support.macros.config
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()

if(TARGET libc.src.__support.alloc.${LIBC_TARGET_OS}.page)
  add_object_library(
    page
    ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.page
  )

  add_object_library(
    arena
    SRCS
      arena.cpp
    HDRS
      arena.h
    COMPILE_OPTIONS
      -DLIBC_PAGE_SIZE=${LIBC_CONF_PAGE_SIZE}
    DEPENDS
      .base
      .page
      libc.src.string.memmove
      libc.src.unistd.getpagesize
  )
endif()

if(NOT ${LIBC_CONF_ALLOC_TYPE} MATCHES "LIBC_ALLOC_TYPE_SCUDO" AND NOT ${LIBC_CONF_ALLOC_TYPE} MATCHES "LIBC_ALLOC_TYPE_EXTERN")
  string(TOLOWER ${LIBC_CONF_ALLOC_TYPE} LIBC_CONF_ALLOC_TYPE_NAME)
  string(REPLACE "libc_alloc_type_" "" LIBC_CONF_ALLOC_TYPE_NAME "${LIBC_CONF_ALLOC_TYPE_NAME}")
  add_object_library(
    alloc
    SRCS
      alloc.cpp
    HDRS
      alloc.h
    COMPILE_OPTIONS
      -DLIBC_CONF_ALLOC_TYPE=${LIBC_CONF_ALLOC_TYPE_NAME}
    DEPENDS
      .${LIBC_CONF_ALLOC_TYPE_NAME}
  )
endif()
