if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()

add_header_library(
  units
  HDRS
    units.h
  DEPENDS
    libc.src.__support.common
    libc.hdr.types.time_t
)
