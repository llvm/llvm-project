if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  __llvm_libc_makedev
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__llvm_libc_makedev
)

add_entrypoint_object(
  __llvm_libc_major
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__llvm_libc_major
)

add_entrypoint_object(
  __llvm_libc_minor
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.__llvm_libc_minor
)
