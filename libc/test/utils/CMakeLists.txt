add_subdirectory(FPUtil)
add_subdirectory(UnitTest)

if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

add_subdirectory(tools)
