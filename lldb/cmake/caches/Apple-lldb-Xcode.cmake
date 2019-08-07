include(${CMAKE_CURRENT_LIST_DIR}/Apple-lldb-base.cmake)

set(CMAKE_GENERATOR Xcode CACHE STRING "")
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.11 CACHE STRING "")
set(CMAKE_XCODE_GENERATE_SCHEME ON CACHE BOOL "")

set(LLDB_BUILD_FRAMEWORK ON CACHE BOOL "")

# Print a warning with instructions, if we
# build with Xcode and didn't use this cache.
set(LLDB_EXPLICIT_XCODE_CACHE_USED ON CACHE INTERNAL "")
