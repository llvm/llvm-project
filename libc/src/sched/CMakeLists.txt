if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  sched_getaffinity
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sched_getaffinity
)

add_entrypoint_object(
  sched_setaffinity
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sched_setaffinity
)

add_entrypoint_object(
  sched_getcpucount
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sched_getcpucount
)
