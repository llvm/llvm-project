add_subdirectory(hdrgen)

if(LIBC_INCLUDE_SIZE_CHECKS)
  add_subdirectory(size_check)
endif()

if(LLVM_INCLUDE_TESTS)
  add_subdirectory(MPFRWrapper)
  add_subdirectory(MPCWrapper)
endif()
