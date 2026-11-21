if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  closelog
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.closelog
)

add_entrypoint_object(
  openlog
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.openlog
)

add_entrypoint_object(
  setlogmask
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.setlogmask
)

add_entrypoint_object(
  syslog
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.syslog
)

add_entrypoint_object(
  vsyslog
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.vsyslog
)
