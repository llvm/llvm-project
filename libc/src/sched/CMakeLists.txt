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
  sched_yield
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sched_yield
)

add_entrypoint_object(
  sched_setparam
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sched_setparam
)

add_entrypoint_object(
  sched_getparam
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sched_getparam
)

add_entrypoint_object(
  sched_setscheduler
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sched_setscheduler
)

add_entrypoint_object(
  sched_getscheduler
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sched_getscheduler
)

add_entrypoint_object(
  sched_get_priority_min
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sched_get_priority_min
)

add_entrypoint_object(
  sched_get_priority_max
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sched_get_priority_max
)

add_entrypoint_object(
  sched_rr_get_interval
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sched_rr_get_interval
)

add_entrypoint_object(
  __sched_getcpucount
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__sched_getcpucount
)
