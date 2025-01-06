add_header_library(
  units
  HDRS
    units.h
  DEPENDS
    libc.src.__support.common
    libc.hdr.types.time_t
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
else()
  return()
endif()

add_object_library(
  clock_gettime
  ALIAS
  DEPENDS
    libc.src.__support.time.${LIBC_TARGET_OS}.clock_gettime
)
