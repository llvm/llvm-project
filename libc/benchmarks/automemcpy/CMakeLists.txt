if(NOT LIBC_BUILD_AUTOMEMCPY)
  return ()
endif()

if(NOT LLVM_WITH_Z3)
  MESSAGE(FATAL_ERROR "Building llvm-libc automemcpy requires Z3")
endif()

set(LIBC_AUTOMEMCPY_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(lib)
add_subdirectory(unittests)
