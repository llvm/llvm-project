#===============================================================================
# Define targets for linking against the selected C library
#
# After including this file, the following targets are defined:
# - libcxx-libc-headers: An interface target that allows getting access to the
#                        headers of the selected C library.
# - libcxx-libc-shared: A target representing the selected shared C library.
# - libcxx-libm-shared: A target representing the selected shared C math library.
# - libcxx-libc-static: A target representing the selected static C library.
# - libcxx-libm-static: A target representing the selected static C math library.
#===============================================================================

# Link against a system-provided libc
if (LIBCXX_LIBC STREQUAL "system")
  add_library(libcxx-libc-headers INTERFACE)

  add_library(libcxx-libc-static INTERFACE)
  add_library(libcxx-libm-static INTERFACE)

  add_library(libcxx-libc-shared INTERFACE)
  add_library(libcxx-libm-shared INTERFACE)

# Link against the in-tree LLVM libc
elseif (LIBCXX_LIBC STREQUAL "llvm-libc")
  add_library(libcxx-libc-headers INTERFACE)
  target_link_libraries(libcxx-libc-headers INTERFACE libc-headers)

  add_library(libcxx-libc-static ALIAS libc)
  add_library(libcxx-libm-static ALIAS libm)

  # TODO: There's no support for building LLVM libc as a shared library yet.
  add_library(libcxx-libc-shared INTERFACE)
  add_library(libcxx-libm-shared INTERFACE)
endif()
