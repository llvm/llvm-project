if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  dl_iterate_phdr
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.dl_iterate_phdr
)
