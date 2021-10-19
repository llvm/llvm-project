add_subdirectory(CPP)
add_subdirectory(MPFRWrapper)
add_subdirectory(testutils)
add_subdirectory(UnitTest)

if(LLVM_LIBC_FULL_BUILD)
  add_subdirectory(tools)
endif()
