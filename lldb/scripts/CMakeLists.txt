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

if(LLDB_BUILD_FRAMEWORK)
  set(framework_arg --framework --target-platform Darwin)
endif()

find_package(SWIG REQUIRED)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lldb.py
  DEPENDS ${SWIG_SOURCES}
  DEPENDS ${SWIG_INTERFACES}
  DEPENDS ${SWIG_HEADERS}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Python/prepare_binding_Python.py
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/prepare_bindings.py
      ${framework_arg}
      --srcRoot=${LLDB_SOURCE_DIR}
      --targetDir=${CMAKE_CURRENT_BINARY_DIR}
      --cfgBldDir=${CMAKE_CURRENT_BINARY_DIR}
      --prefix=${CMAKE_BINARY_DIR}
      --swigExecutable=${SWIG_EXECUTABLE}
  VERBATIM
  COMMENT "Python script building LLDB Python wrapper")

add_custom_target(swig_wrapper ALL DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/lldb.py
)

if(NOT LLDB_BUILD_FRAMEWORK)
  if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(swig_python_subdir site-packages)
  else()
    set(swig_python_subdir python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
  endif()

  set(SWIG_PYTHON_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/${swig_python_subdir})
  set(SWIG_INSTALL_DIR lib${LLVM_LIBDIR_SUFFIX})

  # Install the LLDB python module
  install(DIRECTORY ${SWIG_PYTHON_DIR} DESTINATION ${SWIG_INSTALL_DIR})
endif()
