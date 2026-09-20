if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  getcpu
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.getcpu
)

add_entrypoint_object(
  sched_getaffinity
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sched_getaffinity
)

add_entrypoint_object(
  sched_getcpu
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sched_getcpu
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
  __sched_andcpuset
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__sched_andcpuset
)

add_entrypoint_object(
  __sched_clrcpuset
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__sched_clrcpuset
)

add_entrypoint_object(
  __sched_cpualloc
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__sched_cpualloc
)

add_entrypoint_object(
  __sched_cpuequal
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__sched_cpuequal
)

add_entrypoint_object(
  __sched_cpufree
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__sched_cpufree
)

add_entrypoint_object(
  __sched_getcpucount
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__sched_getcpucount
)

add_entrypoint_object(
  __sched_setcpuzero
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__sched_setcpuzero
)

add_entrypoint_object(
  __sched_setcpuset
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__sched_setcpuset
)

add_entrypoint_object(
  __sched_getcpuisset
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__sched_getcpuisset
)

add_entrypoint_object(
  __sched_orcpuset
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__sched_orcpuset
)

add_entrypoint_object(
  __sched_xorcpuset
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__sched_xorcpuset
)
