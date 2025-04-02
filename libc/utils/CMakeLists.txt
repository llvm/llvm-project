add_subdirectory(hdrgen)

if(LLVM_INCLUDE_TESTS)
  add_subdirectory(MPFRWrapper)
  add_subdirectory(MPCWrapper)
endif()
