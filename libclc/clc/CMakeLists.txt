# Add the target specific files first so they can override the generic fallback.
if(LIBCLC_ARCH_DIR AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/${LIBCLC_ARCH_DIR})
  add_subdirectory(lib/${LIBCLC_ARCH_DIR})
endif()
add_subdirectory(lib/generic)
