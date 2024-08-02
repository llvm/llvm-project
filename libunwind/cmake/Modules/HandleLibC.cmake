#===============================================================================
# Define targets for linking against the selected C library
#
# After including this file, the following targets are defined:
# - libunwind-libc-headers: An interface target that allows getting access to the
#                        headers of the selected C library.
# - libunwind-libc-shared: A target representing the selected shared C library.
# - libunwind-libc-static: A target representing the selected static C library.
#===============================================================================

# Link against a system-provided libc
if (LIBUNWIND_LIBC STREQUAL "system")
  add_library(libunwind-libc-headers INTERFACE)

  add_library(libunwind-libc-static INTERFACE)
  add_library(libunwind-libc-shared INTERFACE)

# Link against the in-tree LLVM libc
elseif (LIBUNWIND_LIBC STREQUAL "llvm-libc")
  add_library(libunwind-libc-headers INTERFACE)
  target_link_libraries(libunwind-libc-headers INTERFACE libc-headers)
  if(CXX_SUPPORTS_NOSTDLIBINC_FLAG)
    target_compile_options(libunwind-libc-headers INTERFACE "-nostdlibinc")
  endif()

  add_library(libunwind-libc-static INTERFACE)
  if (TARGET libc)
    target_link_libraries(libunwind-libc-static INTERFACE libc)
  endif()
  if (TARGET libm)
    target_link_libraries(libunwind-libc-static INTERFACE libm)
  endif()
  if (CXX_SUPPORTS_NOLIBC_FLAG)
    target_link_options(libunwind-libc-static INTERFACE "-nolibc")
  endif()

  # TODO: There's no support for building LLVM libc as a shared library yet.
  add_library(libunwind-libc-shared INTERFACE)
endif()
