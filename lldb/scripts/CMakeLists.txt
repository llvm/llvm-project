file(GLOB SWIG_INTERFACES interface/*.i)
file(GLOB_RECURSE SWIG_SOURCES *.swig)
file(GLOB SWIG_HEADERS
  ${LLDB_SOURCE_DIR}/include/lldb/API/*.h
  ${LLDB_SOURCE_DIR}/include/lldb/*.h
)
file(GLOB SWIG_PRIVATE_HEADERS
  ${LLDB_SOURCE_DIR}/include/lldb/lldb-private*.h
)
foreach(private_header ${SWIG_PRIVATE_HEADERS})
  list(REMOVE_ITEM SWIG_HEADERS ${private_header})
endforeach()

if(LLDB_BUILD_FRAMEWORK)
  set(framework_arg --framework --target-platform Darwin)
endif()

find_package(SWIG REQUIRED)
set(SWIG_MIN_VERSION "2.0.0")
if (${SWIG_VERSION} VERSION_LESS ${SWIG_MIN_VERSION})
  message(FATAL_ERROR "LLDB requires swig ${SWIG_MIN_VERSION}, your version is ${SWIG_VERSION}.")
endif()

if(APPLE)
  set(DARWIN_EXTRAS "-D__APPLE__")
else()
  set(DARWIN_EXTRAS "")
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lldb.py
  DEPENDS ${SWIG_SOURCES}
  DEPENDS ${SWIG_INTERFACES}
  DEPENDS ${SWIG_HEADERS}
  COMMAND ${SWIG_EXECUTABLE} 
      -c++
      -shadow
      -python
      -features autodoc
      -threads
      -I${LLDB_SOURCE_DIR}/include
      -I${CMAKE_CURRENT_SOURCE_DIR}
      -D__STDC_LIMIT_MACROS
      -D__STDC_CONSTANT_MACROS
      ${DARWIN_EXTRAS}
      -outdir ${CMAKE_CURRENT_BINARY_DIR}
      -o ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
      ${LLDB_SOURCE_DIR}/scripts/lldb.swig
  VERBATIM
  COMMENT "Builds LLDB Python wrapper")

add_custom_target(swig_wrapper ALL DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/lldb.py
)

if(NOT LLDB_BUILD_FRAMEWORK)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE}
        -c "import distutils.sysconfig, sys; print(distutils.sysconfig.get_python_lib(True, False, sys.argv[1]))"
        ${CMAKE_BINARY_DIR}
    OUTPUT_VARIABLE SWIG_PYTHON_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE}
        -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_lib(True, False, ''))"
    OUTPUT_VARIABLE SWIG_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # Install the LLDB python module
  add_custom_target(lldb-python-scripts)
  add_dependencies(lldb-python-scripts finish_swig)
  install(DIRECTORY ${SWIG_PYTHON_DIR}/
    DESTINATION ${SWIG_INSTALL_DIR}
    COMPONENT lldb-scripts)
  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-lldb-python-scripts
                             COMPONENT lldb-python-scripts
                             DEPENDS lldb-python-scripts)
  endif()
endif()
