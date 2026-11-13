add_header_library(
  assert_fail_hdrs
  HDRS
    assert_fail.h
  DEPENDS
    libc.src.__support.macros.config
)

add_header_library(
  assert_fail_utils
  HDRS
    assert_fail_utils.h
  DEPENDS
    libc.src.__support.integer_to_string
    libc.src.__support.macros.attributes
    libc.src.__support.macros.config
)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  return()
endif()
add_subdirectory(${LIBC_TARGET_OS})

set(target_os_util libc.src.__support.OSUtil.${LIBC_TARGET_OS}.${LIBC_TARGET_OS}_util)
if(TARGET ${target_os_util})
  add_object_library(
    osutil
    ALIAS
      ${target_os_util}
    DEPENDS
      ${target_os_util}
  )
endif()

set(target_assert_fail libc.src.__support.OSUtil.${LIBC_TARGET_OS}.assert_fail)
if(TARGET ${target_assert_fail})
  add_object_library(
    assert_fail
    ALIAS
      ${target_assert_fail}
    DEPENDS
      ${target_assert_fail}
  )
endif()

set(target_path libc.src.__support.OSUtil.${LIBC_TARGET_OS}.path)
if(TARGET ${target_path})
  add_header_library(
    path
    HDRS
      path.h
    DEPENDS
      ${target_path}
  )
endif()
