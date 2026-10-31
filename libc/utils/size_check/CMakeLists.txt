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

# Linked size probes need the full build archives. The archive targets are
# defined later (in libc/lib), so mirror the conditions they are created with.
_libc_size_check_find_tool(LIBC_SIZE_CHECK_READELF llvm-readelf)
set(probe_archives "")
if(LLVM_LIBC_FULL_BUILD AND NOT LLVM_LIBC_ENABLE_MINOR_VARIANT)
  if(TARGET_LIBM_ENTRYPOINTS)
    list(APPEND probe_archives libm)
  endif()
  if(TARGET_LIBC_ENTRYPOINTS)
    list(APPEND probe_archives libc)
  endif()
endif()
if(NOT probe_archives)
  message(STATUS "libc size probes need a full build; skipping them.")
elseif(NOT LIBC_SIZE_CHECK_READELF)
  message(WARNING "libc size probes need llvm-readelf; set "
    "LIBC_SIZE_CHECK_READELF to enable them.")
else()
  add_libc_size_probes(
    libc-size-probes-math
    PROBES ${CMAKE_CURRENT_SOURCE_DIR}/probes/math.txt
    MANIFEST ${CMAKE_CURRENT_BINARY_DIR}/libc-size-check-math-objects.manifest
    ARCHIVE_TARGETS ${probe_archives}
  )
  add_dependencies(check-libc-size libc-size-probes-math)
endif()
