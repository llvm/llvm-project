add_subdirectory(FPUtil)
add_subdirectory(CPP)

if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

add_subdirectory(tools)
