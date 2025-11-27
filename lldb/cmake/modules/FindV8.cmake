#.rst:
# FindV8
# ------
#
# Find V8 JavaScript engine
#
# This module will search for V8 in standard system locations, or use
# user-specified paths. Users can override the search by setting:
#   -DV8_INCLUDE_DIR=/path/to/v8/include
#   -DV8_LIBRARIES=/path/to/libv8.so (or libv8_monolith.a)
#
# The module defines:
#   V8_FOUND - System has V8
#   V8_INCLUDE_DIR - V8 include directory
#   V8_LIBRARIES - V8 libraries to link against

if(V8_LIBRARIES AND V8_INCLUDE_DIR)
  set(V8_FOUND TRUE)
  if(NOT V8_FIND_QUIETLY)
    message(STATUS "Found V8: ${V8_INCLUDE_DIR}")
    message(STATUS "Found V8 library: ${V8_LIBRARIES}")
    set(V8_FIND_QUIETLY TRUE CACHE BOOL "Suppress repeated V8 find messages" FORCE)
  endif()
else()
  # Try to find system V8
  find_path(V8_INCLUDE_DIR
    NAMES v8.h
    PATHS
      # Standard system locations
      /usr/include
      /usr/local/include
      /opt/v8/include
      # Homebrew on macOS
      /opt/homebrew/include
      /usr/local/opt/v8/include
    PATH_SUFFIXES
      v8
    DOC "V8 include directory"
  )

  find_library(V8_LIBRARIES
    NAMES v8_monolith v8 v8_libbase v8_libplatform
    PATHS
      # Standard system locations
      /usr/lib
      /usr/local/lib
      /opt/v8/lib
      # Homebrew on macOS
      /opt/homebrew/lib
      /usr/local/opt/v8/lib
    DOC "V8 library"
  )

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(V8
                                    FOUND_VAR
                                      V8_FOUND
                                    REQUIRED_VARS
                                      V8_INCLUDE_DIR
                                      V8_LIBRARIES)

  if(V8_FOUND)
    mark_as_advanced(V8_LIBRARIES V8_INCLUDE_DIR)
    message(STATUS "Found V8: ${V8_INCLUDE_DIR}")
    if(V8_LIBRARIES)
      message(STATUS "Found V8 library: ${V8_LIBRARIES}")
    else()
      message(STATUS "V8 headers found (library may need to be built or specified manually)")
    endif()
  endif()
endif()
