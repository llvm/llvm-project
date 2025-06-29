if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()

add_subdirectory(threads)
if(LIBC_TARGET_OS_IS_GPU)
  add_subdirectory(GPU)
endif()
