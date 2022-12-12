add_subdirectory(MPFRWrapper)
add_subdirectory(testutils)
add_subdirectory(UnitTest)

if(LLVM_LIBC_FULL_BUILD AND NOT LIBC_TARGET_ARCHITECTURE_IS_GPU)
  add_subdirectory(IntegrationTest)
  add_subdirectory(tools)
endif()
