if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_subdirectory(generic)

if(TARGET libc.src.compiler.${LIBC_TARGET_OS}.__stack_chk_fail)
  set(stack_chk_fail_dep libc.src.compiler.${LIBC_TARGET_OS}.__stack_chk_fail)
else()
  set(stack_chk_fail_dep libc.src.compiler.generic.__stack_chk_fail)
endif()

add_entrypoint_object(
  __stack_chk_fail
  ALIAS
  DEPENDS
    ${stack_chk_fail_dep}
)

if(TARGET libc.src.compiler.${LIBC_TARGET_OS}.__stack_chk_guard)
  set(stack_chk_guard_dep libc.src.compiler.${LIBC_TARGET_OS}.__stack_chk_guard)
else()
  set(stack_chk_guard_dep libc.src.compiler.generic.__stack_chk_guard)
endif()

add_entrypoint_object(
  __stack_chk_guard
  ALIAS
  DEPENDS
    ${stack_chk_guard_dep}
)
