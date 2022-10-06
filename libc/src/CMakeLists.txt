add_subdirectory(__support)

add_subdirectory(ctype)
add_subdirectory(errno)
add_subdirectory(fenv)
add_subdirectory(inttypes)
add_subdirectory(math)
add_subdirectory(string)
add_subdirectory(stdlib)
add_subdirectory(stdio)

if(${LIBC_TARGET_OS} STREQUAL "linux")
  add_subdirectory(dirent)
  add_subdirectory(fcntl)
  add_subdirectory(pthread)
  add_subdirectory(sched)
  add_subdirectory(sys)
  add_subdirectory(unistd)
endif()

if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

# The signal API is currently disabled as signal.h is incorrect.
# since assert uses the signal API, we disable assert also.
# add_subdirectory(assert)
add_subdirectory(signal)
add_subdirectory(threads)
add_subdirectory(time)
