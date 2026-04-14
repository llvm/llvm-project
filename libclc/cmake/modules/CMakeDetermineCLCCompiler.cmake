if(NOT CMAKE_CLC_COMPILER)
  if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
      "The CLC language requires the C compiler (CMAKE_C_COMPILER) to be "
      "Clang, but CMAKE_C_COMPILER_ID is '${CMAKE_C_COMPILER_ID}'.")
  endif()
  set(CMAKE_CLC_COMPILER "${CMAKE_C_COMPILER}" CACHE FILEPATH "CLC compiler")
endif()

mark_as_advanced(CMAKE_CLC_COMPILER)

set(CMAKE_CLC_COMPILER_ID "Clang")
set(CMAKE_CLC_COMPILER_ID_RUN TRUE)

configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeCLCCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeCLCCompiler.cmake @ONLY)

set(CMAKE_CLC_COMPILER_ENV_VAR "CLC")
