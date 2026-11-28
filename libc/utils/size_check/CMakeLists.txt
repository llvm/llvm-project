# Binary size checks for libc entrypoints. See README.md.

if(LIBC_TARGET_OS_IS_DARWIN OR LIBC_TARGET_OS_IS_WINDOWS OR
  LIBC_TARGET_OS_IS_GPU OR LIBC_TARGET_OS_IS_UEFI)
  message(STATUS "libc size checks are only supported for ELF targets; "
    "skipping them for ${LIBC_TARGET_TRIPLE}.")
  return()
endif()

_libc_size_check_find_tool(LIBC_SIZE_CHECK_NM llvm-nm)
_libc_size_check_find_tool(LIBC_SIZE_CHECK_SIZE llvm-size)
if(NOT LIBC_SIZE_CHECK_NM OR NOT LIBC_SIZE_CHECK_SIZE)
  message(WARNING "libc size checks need llvm-nm and llvm-size; set "
    "LIBC_SIZE_CHECK_NM and LIBC_SIZE_CHECK_SIZE to enable them.")
  return()
endif()

set(LIBC_SIZE_CHECK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/libc_size_check.py)

add_custom_target(
  check-libc-size-unittests
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/libc_size_check_test.py
  COMMENT "Running the libc size checker unit tests"
  USES_TERMINAL
)

add_libc_object_size_check(
  libc-size-check-math-objects
  ENTRYPOINT_REGEX "^libc\\.src\\.math\\."
  RULES ${CMAKE_CURRENT_SOURCE_DIR}/rules/math.json
)

add_custom_target(check-libc-size)
add_dependencies(
  check-libc-size
  check-libc-size-unittests
  libc-size-check-math-objects
)
