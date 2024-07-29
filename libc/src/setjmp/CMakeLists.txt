if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
endif()

add_header_library(
  checksum
  HDRS
    checksum.h
  DEPENDS
    libc.src.__support.hash
    libc.src.stdlib.abort
    libc.src.unistd.write
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
