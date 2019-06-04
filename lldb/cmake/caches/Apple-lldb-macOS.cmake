include(${CMAKE_CURRENT_LIST_DIR}/Apple-lldb-base.cmake)

set(LLDB_BUILD_FRAMEWORK ON CACHE BOOL "")
set(LLDB_NO_INSTALL_DEFAULT_RPATH ON CACHE BOOL "")

# Set the install prefix to the default install location on the enduser machine.
# If the location is not writeable on the build machine, specify another prefix
# in the DESTDIR environment variable, e.g.: DESTDIR=/tmp ninja install
set(CMAKE_INSTALL_PREFIX /Applications/Xcode.app/Contents/Developer/usr CACHE STRING "")

# Choose the install location for LLDB.framework so that it matches the
# INSTALL_RPATH of the lldb driver. It's either absolute or relative to
# CMAKE_INSTALL_PREFIX. In any case, DESTDIR will be an extra prefix.
set(LLDB_FRAMEWORK_INSTALL_DIR /Applications/Xcode.app/Contents/SharedFrameworks CACHE STRING "")

# Release builds may change these:
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.11 CACHE STRING "")
set(LLDB_USE_SYSTEM_DEBUGSERVER ON CACHE BOOL "")
set(LLVM_EXTERNALIZE_DEBUGINFO OFF CACHE BOOL "")
