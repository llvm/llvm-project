#===============================================================================
# Define targets for linking against the selected C library
#
# After including this file, the following targets are defined:
# - libcxxabi-libc-headers: An interface target that allows getting access to the
#                        headers of the selected C library.
# - libcxxabi-libc-shared: A target representing the selected shared C library.
# - libcxxabi-libc-static: A target representing the selected static C library.
#===============================================================================

# Link against a system-provided libc
if (LIBCXXABI_LIBC STREQUAL "system")
  add_library(libcxxabi-libc-headers INTERFACE)

  add_library(libcxxabi-libc-static INTERFACE)
  add_library(libcxxabi-libc-shared INTERFACE)

# Link against the in-tree LLVM libc
elseif (LIBCXXABI_LIBC STREQUAL "llvm-libc")
  add_library(libcxxabi-libc-headers INTERFACE)
  target_link_libraries(libcxxabi-libc-headers INTERFACE libc-headers)
  if(CXX_SUPPORTS_NOSTDLIBINC_FLAG)
    target_compile_options(libcxxabi-libc-headers INTERFACE "-nostdlibinc")
  endif()

  add_library(libcxxabi-libc-static INTERFACE)
  if (TARGET libc)
    target_link_libraries(libcxxabi-libc-static INTERFACE libc)
  endif()
  if (TARGET libm)
    target_link_libraries(libcxxabi-libc-static INTERFACE libm)
  endif()
  if (CXX_SUPPORTS_NOLIBC_FLAG)
    target_link_options(libcxxabi-libc-static INTERFACE "-nolibc")
  endif()

  # TODO: There's no support for building LLVM libc as a shared library yet.
  add_library(libcxxabi-libc-shared INTERFACE)
endif()
