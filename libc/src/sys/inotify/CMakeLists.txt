if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  inotify_init
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.inotify_init
)

add_entrypoint_object(
  inotify_init1
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.inotify_init1
)

add_entrypoint_object(
  inotify_add_watch
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.inotify_add_watch
)

add_entrypoint_object(
  inotify_rm_watch
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.inotify_rm_watch
)
