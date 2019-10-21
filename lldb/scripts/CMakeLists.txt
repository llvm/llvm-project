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

if(${LLDB_USE_STATIC_BINDINGS})
  set(PREPARE_BINDINGS_ARGS
    --use-static-binding)
else()
  find_package(SWIG)
  if(${SWIG_FOUND})
    set(SWIG_MIN_VERSION "2.0.0")
    if (${SWIG_VERSION} VERSION_LESS ${SWIG_MIN_VERSION})
      message(FATAL_ERROR "LLDB requires swig ${SWIG_MIN_VERSION}, your version is ${SWIG_VERSION}.")
    endif()
    set(PREPARE_BINDINGS_ARGS
      "--swig-executable=${SWIG_EXECUTABLE}")
  else()
    message(FATAL_ERROR "LLDB requires swig ${SWIG_MIN_VERSION}. To use the static bindings instead use -DLLDB_USE_STATIC_BINDINGS=ON")
  endif()
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

