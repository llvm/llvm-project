file(GLOB SWIG_INTERFACES interface/*.i)
file(GLOB_RECURSE SWIG_SOURCES *.swig)
set(SWIG_HEADERS
  ${LLDB_SOURCE_DIR}/include/lldb/API/SBDefines.h
  ${LLDB_SOURCE_DIR}/include/lldb/lldb-defines.h
  ${LLDB_SOURCE_DIR}/include/lldb/lldb-enumerations.h
  ${LLDB_SOURCE_DIR}/include/lldb/lldb-forward.h
  ${LLDB_SOURCE_DIR}/include/lldb/lldb-types.h
  ${LLDB_SOURCE_DIR}/include/lldb/lldb-versioning.h
)

include(FindPythonInterp)

if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(SWIG_PYTHON_DIR
    ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
else()
  set(SWIG_PYTHON_DIR ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/site-packages)
endif()

set(SWIG_INSTALL_DIR lib${LLVM_LIBDIR_SUFFIX})

# Generating the LLDB framework correctly is a bit complicated because the
# framework depends on the swig output.
if(LLDB_BUILD_FRAMEWORK)
  set(framework_arg --framework --target-platform Darwin)
  set(SWIG_PYTHON_DIR
    ${LLDB_PYTHON_TARGET_DIR}/${LLDB_FRAMEWORK_RESOURCE_DIR}/Python)
  set(SWIG_INSTALL_DIR
    ${LLDB_FRAMEWORK_INSTALL_DIR}/${LLDB_FRAMEWORK_RESOURCE_DIR})
endif()

get_filename_component(CFGBLDDIR ${LLDB_WRAP_PYTHON} DIRECTORY)

find_package(SWIG REQUIRED)
add_custom_command(
  OUTPUT ${LLDB_WRAP_PYTHON}
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lldb.py
  DEPENDS ${SWIG_SOURCES}
  DEPENDS ${SWIG_INTERFACES}
  DEPENDS ${SWIG_HEADERS}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Python/prepare_binding_Python.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Python/modify-python-lldb.py
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/prepare_bindings.py
      ${framework_arg}
      --srcRoot=${LLDB_SOURCE_DIR}
      --targetDir=${LLDB_PYTHON_TARGET_DIR}
      --cfgBldDir=${CFGBLDDIR}
      --prefix=${CMAKE_BINARY_DIR}
      --swigExecutable=${SWIG_EXECUTABLE}
  VERBATIM
  COMMENT "Python script building LLDB Python wrapper")
add_custom_target(swig_wrapper ALL DEPENDS ${LLDB_WRAP_PYTHON})

set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/lldb.py PROPERTIES GENERATED 1)


# Install the LLDB python module
install(DIRECTORY ${SWIG_PYTHON_DIR} DESTINATION ${SWIG_INSTALL_DIR})

# build Python modules
add_subdirectory(Python/modules)
