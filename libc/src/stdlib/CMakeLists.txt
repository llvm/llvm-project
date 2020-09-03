if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  _Exit
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}._Exit
)

add_entrypoint_object(
  abort
  SRCS
    abort.cpp
  HDRS
    abort.h
  DEPENDS
    libc.include.stdlib
    libc.src.signal.raise
    ._Exit
)
