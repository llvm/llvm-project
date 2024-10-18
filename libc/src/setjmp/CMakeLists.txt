if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
endif()

if (LIBC_CONF_SETJMP_ENABLE_FORTIFICATION)
  set(checksum_flags "-DLIBC_COPT_SETJMP_ENABLE_FORTIFICATION=1")
else()
  set(checksum_flags "-DLIBC_COPT_SETJMP_ENABLE_FORTIFICATION=0")
endif()


add_header_library(
  checksum
  HDRS
    checksum.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.OSUtil.osutil
  COMPILE_OPTIONS
    ${checksum_flags}
)

add_entrypoint_object(
  setjmp
  ALIAS
  DEPENDS
    .${LIBC_TARGET_ARCHITECTURE}.setjmp
)

add_entrypoint_object(
  longjmp
  ALIAS
  DEPENDS
    .${LIBC_TARGET_ARCHITECTURE}.longjmp
)
