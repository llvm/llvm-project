# This CMake script installs the LLDB python module from the build directory
# to the install directory.

# FIXME: if a non-standard version of python is requested, the cmake macro
# below will need Python_ADDITIONAL_VERSIONS set in order to find it.
include(FindPythonInterp)

SET(PYTHON_DIRECTORY python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})

SET(lldb_module_src ${CMAKE_CURRENT_BINARY_DIR}/lib/${PYTHON_DIRECTORY})
SET(lldb_module_dest ${CMAKE_INSTALL_PREFIX}/lib)

MESSAGE(STATUS "Installing LLDB python module from: ${lldb_module_src} to ${lldb_module_dest}")
FILE(COPY "${lldb_module_src}" DESTINATION "${lldb_module_dest}")
