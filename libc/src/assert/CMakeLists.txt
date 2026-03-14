if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
else()
  add_subdirectory(generic)
endif()

if(TARGET libc.src.assert.${LIBC_TARGET_OS}.__assert_fail)
  set(assert_fail_dep libc.src.assert.${LIBC_TARGET_OS}.__assert_fail)
else()
  set(assert_fail_dep libc.src.assert.generic.__assert_fail)
endif()

add_entrypoint_object(
  __assert_fail
  ALIAS
  DEPENDS
    ${assert_fail_dep}
)
