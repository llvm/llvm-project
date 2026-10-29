# The startup dynamic linker. It is only built when there are shared objects
# for it to load.
if(NOT LIBC_ENABLE_SHARED)
  return()
endif()
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  message(STATUS "Skipping the loader for ${LIBC_TARGET_OS}")
  return()
endif()
add_subdirectory(${LIBC_TARGET_OS})
