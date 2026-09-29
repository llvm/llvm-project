# Mimick `GNUInstallDirs` for one more install directory, the one where
# project's installed cmake subdirs go.

# These functions are internal functions vendored in from GNUInstallDirs (with
# new names), so we don't depend on unstable implementation details. They are
# also simplified to only handle the cases we need.
#
# The purpose would appear to be making `CACHE PATH` vars in a way that
# bypasses the legacy oddity that `-D<PATH>` gets canonicalized, despite
# non-canonical `CACHE PATH`s being perfectly valid.

macro(_GNUInstallPackageDir_cache_convert_to_path var description)
  get_property(_GNUInstallPackageDir_cache_type CACHE ${var} PROPERTY TYPE)
  if(_GNUInstallPackageDir_cache_type STREQUAL "UNINITIALIZED")
    file(TO_CMAKE_PATH "${${var}}" _GNUInstallPackageDir_cmakepath)
    set_property(CACHE ${var} PROPERTY TYPE PATH)
    set_property(CACHE ${var} PROPERTY VALUE "${_GNUInstallPackageDir_cmakepath}")
    set_property(CACHE ${var} PROPERTY HELPSTRING "${description}")
    unset(_GNUInstallPackageDir_cmakepath)
  endif()
  unset(_GNUInstallPackageDir_cache_type)
endmacro()

# Create a cache variable with default for a path.
macro(_GNUInstallPackageDir_cache_path var default description)
  if(NOT DEFINED ${var})
    set(${var} "${default}" CACHE PATH "${description}")
  endif()
  _GNUInstallPackageDir_cache_convert_to_path("${var}" "${description}")
endmacro()

_GNUInstallPackageDir_cache_path(CMAKE_INSTALL_PACKAGEDIR "lib${LLVM_LIBDIR_SUFFIX}/cmake"
  "Directories containing installed CMake modules (lib/cmake)")
