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

find_package(SWIG)
if(${SWIG_FOUND})
  set(SWIG_MIN_VERSION "2.0.0")
  if (${SWIG_VERSION} VERSION_LESS ${SWIG_MIN_VERSION})
    message(FATAL_ERROR "LLDB requires swig ${SWIG_MIN_VERSION}, your version is ${SWIG_VERSION}.")
  endif()
  set(PREPARE_BINDINGS_ARGS
    "--swig-executable=${SWIG_EXECUTABLE}")
elseif(${LLDB_ALLOW_STATIC_BINDINGS})
  set(PREPARE_BINDINGS_ARGS
    --use-static-binding)
else()
  message(FATAL_ERROR "swig not found and static bindings not permitted - install swig or specify -DLLDB_ALLOW_STATIC_BINDINGS=1")
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
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Python/prepare_binding_Python.py
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/prepare_bindings.py
      ${framework_arg}
      --srcRoot=${LLDB_SOURCE_DIR}
      --targetDir=${CMAKE_CURRENT_BINARY_DIR}
      --cfgBldDir=${CMAKE_CURRENT_BINARY_DIR}
      --prefix=${CMAKE_BINARY_DIR}
      ${PREPARE_BINDINGS_ARGS}
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
  install(DIRECTORY ${SWIG_PYTHON_DIR}/ DESTINATION ${SWIG_INSTALL_DIR})
endif()
