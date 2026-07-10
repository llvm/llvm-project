if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  add_subdirectory(${LIBC_TARGET_ARCHITECTURE})
endif()

add_entrypoint_object(
  getcontext
  ALIAS
  DEPENDS
    .${LIBC_TARGET_ARCHITECTURE}.getcontext
)

add_entrypoint_object(
  setcontext
  ALIAS
  DEPENDS
    .${LIBC_TARGET_ARCHITECTURE}.setcontext
)

