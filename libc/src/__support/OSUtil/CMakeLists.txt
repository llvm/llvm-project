if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  return()
endif()

add_subdirectory(${LIBC_TARGET_OS})
set(target_os_util libc.src.__support.OSUtil.${LIBC_TARGET_OS}.${LIBC_TARGET_OS}_util)
if(NOT TARGET ${target_os_util})
  return()
endif()

add_object_library(
  osutil
  ALIAS
    ${target_os_util}
  DEPENDS
    ${target_os_util}
)

if (LIBC_CONF_ENABLE_PID_CACHE)
  set(libc_copt_enable_pid_cache 1)
else()
  set(libc_copt_enable_pid_cache 0)
endif()

if(TARGET libc.src.__support.OSUtil.${LIBC_TARGET_OS}.pid)
  add_object_library(
    pid
    ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.pid
    COMPILE_OPTIONS
      -DLIBC_COPT_ENABLE_PID_CACHE=${libc_copt_enable_pid_cache}
  )
endif()
