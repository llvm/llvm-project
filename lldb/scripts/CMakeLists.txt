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

find_package(SWIG REQUIRED)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lldb.py
  DEPENDS ${SWIG_SOURCES}
  DEPENDS ${SWIG_INTERFACES}
  DEPENDS ${SWIG_HEADERS}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Python/buildSwigPython.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Python/modify-python-lldb.py
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/prepare_bindings.py "--srcRoot=${LLDB_SOURCE_DIR}" "--targetDir=${CMAKE_CURRENT_BINARY_DIR}" "--cfgBldDir=${CMAKE_CURRENT_BINARY_DIR}" "--prefix=${CMAKE_BINARY_DIR}" "--swigExecutable=${SWIG_EXECUTABLE}"
  COMMENT "Python script building LLDB Python wrapper")
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp PROPERTIES GENERATED 1)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/lldb.py PROPERTIES GENERATED 1)

add_custom_target(swig_wrapper ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
  )

# Install the LLDB python module on all operating systems (except Windows)
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  install(DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}
  DESTINATION lib${LLVM_LIBDIR_SUFFIX})
endif()

# build Python modules
add_subdirectory(Python/modules)
