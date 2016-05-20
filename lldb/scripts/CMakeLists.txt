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

set(PREPARE_BINDINGS_ARGS "--src-root=${LLDB_SOURCE_DIR}" "--target-dir=${CMAKE_CURRENT_BINARY_DIR}" "--config-build-dir=${CMAKE_CURRENT_BINARY_DIR}" "--prefix=${CMAKE_BINARY_DIR}")

include(FindPythonInterp)

find_package(SWIG)
if( ${SWIG_FOUND} )
    set(PREPARE_BINDINGS_ARGS ${PREPARE_BINDINGS_ARGS} "--swig-executable=${SWIG_EXECUTABLE}")
elseif( ${LLDB_ALLOW_STATIC_BINDINGS} )
    set(PREPARE_BINDINGS_ARGS ${PREPARE_BINDINGS_ARGS} --find-swig --allow-static-binding)
else()
    message( FATAL_ERROR "swig not found and static bindings not permitted - install swig or specify -DLLDB_ALLOW_STATIC_BINDINGS=1")
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lldb.py
  DEPENDS ${SWIG_SOURCES}
  DEPENDS ${SWIG_INTERFACES}
  DEPENDS ${SWIG_HEADERS}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Python/prepare_binding_Python.py
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Python/modify-python-lldb.py
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/prepare_bindings.py ${PREPARE_BINDINGS_ARGS}
  COMMENT "Python script building LLDB Python wrapper")

set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp PROPERTIES GENERATED 1)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/lldb.py PROPERTIES GENERATED 1)

add_custom_target(swig_wrapper ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
  )

# Install the LLDB python module on all operating systems (except Windows)
if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  install(DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/python2.7
  DESTINATION lib${LLVM_LIBDIR_SUFFIX})
endif()

# build Python modules
add_subdirectory(Python/modules)
