# FIXME: if a non-standard version of python is requested, the cmake macro
# below will need Python_ADDITIONAL_VERSIONS set in order to find it.
include(FindPythonInterp)
SET(PYTHON_DIRECTORY python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages)

# Build the readline python module
include_directories(${PYTHON_INCLUDE_DIR})
add_library(readline SHARED readline.cpp)
target_link_libraries(readline ${PYTHON_LIBRARY})

if (NOT LLDB_DISABLE_LIBEDIT)
  target_include_directories(readline
                             PRIVATE
                               ${libedit_INCLUDE_DIRS})
  target_link_libraries(readline ${libedit_LIBRARIES})
endif()

# FIXME: the LIBRARY_OUTPUT_PATH seems to be ignored - this is not a
# functional issue for the build dir, though, since the shared lib dir
# for the build is in the python shared library load path, and thus
# python finds it when loading the python readline module.
set_target_properties(readline PROPERTIES
                               PREFIX ""
                               LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/${PYTHON_DIRECTORY})

# Install the readline module.
install(TARGETS readline LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib${LLVM_LIBDIR_SUFFIX}/${PYTHON_DIRECTORY})
