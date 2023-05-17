if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  cfgetispeed
  ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.cfgetispeed
)

add_entrypoint_object(
  cfsetispeed
  ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.cfsetispeed
)

add_entrypoint_object(
  cfgetospeed
  ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.cfgetospeed
)

add_entrypoint_object(
  cfsetospeed
  ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.cfsetospeed
)

add_entrypoint_object(
  tcgetsid
  ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.tcgetsid
)

add_entrypoint_object(
  tcdrain
  ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.tcdrain
)

add_entrypoint_object(
  tcflow
  ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.tcflow
)

add_entrypoint_object(
  tcflush
  ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.tcflush
)

add_entrypoint_object(
  tcsendbreak
  ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.tcsendbreak
)

add_entrypoint_object(
  tcgetattr
  ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.tcgetattr
)

add_entrypoint_object(
  tcsetattr
  ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.tcsetattr
)
