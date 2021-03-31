add_subdirectory(__support)

add_subdirectory(ctype)
add_subdirectory(errno)
add_subdirectory(fenv)
add_subdirectory(math)
add_subdirectory(string)

if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

add_subdirectory(assert)
add_subdirectory(signal)
add_subdirectory(stdio)
add_subdirectory(stdlib)
# TODO: Add this target conditional to the target OS.
add_subdirectory(sys)
add_subdirectory(threads)
add_subdirectory(time)
add_subdirectory(unistd)
