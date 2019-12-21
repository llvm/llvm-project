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

set(SWIG_COMMON_FLAGS
  -c++
  -features autodoc
  -I${LLDB_SOURCE_DIR}/include
  -I${CMAKE_CURRENT_SOURCE_DIR}
  -D__STDC_LIMIT_MACROS
  -D__STDC_CONSTANT_MACROS
  ${DARWIN_EXTRAS}
  -outdir ${CMAKE_CURRENT_BINARY_DIR}
)


if (LLDB_ENABLE_PYTHON)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lldb.py
    DEPENDS ${SWIG_SOURCES}
    DEPENDS ${SWIG_INTERFACES}
    DEPENDS ${SWIG_HEADERS}
    COMMAND ${SWIG_EXECUTABLE}
        ${SWIG_COMMON_FLAGS}
        -c++
        -shadow
        -python
        -threads
        -o ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
        ${LLDB_SOURCE_DIR}/scripts/lldb.swig
    VERBATIM
    COMMENT "Builds LLDB Python wrapper")

  add_custom_target(swig_wrapper ALL DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/lldb.py
  )
endif()

if (LLDB_ENABLE_LUA)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapLua.cpp
    DEPENDS ${SWIG_SOURCES}
    DEPENDS ${SWIG_INTERFACES}
    DEPENDS ${SWIG_HEADERS}
    COMMAND ${SWIG_EXECUTABLE}
        ${SWIG_COMMON_FLAGS}
        -lua
        -w503
        -o ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapLua.cpp
        ${LLDB_SOURCE_DIR}/scripts/lldb_lua.swig
    VERBATIM
    COMMENT "Builds LLDB Lua wrapper")

  add_custom_target(swig_wrapper_lua ALL DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapLua.cpp
  )
endif()
