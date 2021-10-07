if(LLDB_USE_STATIC_BINDINGS)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapLua.cpp
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_CURRENT_SOURCE_DIR}/static-binding/LLDBWrapLua.cpp"
      "${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapLua.cpp")
else()
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapLua.cpp
    DEPENDS ${SWIG_SOURCES}
    DEPENDS ${SWIG_INTERFACES}
    DEPENDS ${SWIG_HEADERS}
    COMMAND ${SWIG_EXECUTABLE}
        ${SWIG_COMMON_FLAGS}
        -I${CMAKE_CURRENT_SOURCE_DIR}
        -lua
        -w503
        -outdir ${CMAKE_CURRENT_BINARY_DIR}
        -o ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapLua.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lua.swig
    VERBATIM
    COMMENT "Building LLDB Lua wrapper")
endif()

add_custom_target(swig_wrapper_lua ALL DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapLua.cpp
)
