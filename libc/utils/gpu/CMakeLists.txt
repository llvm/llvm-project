add_subdirectory(server)
if(LIBC_TARGET_OS_IS_GPU)
  add_subdirectory(loader)
endif()
