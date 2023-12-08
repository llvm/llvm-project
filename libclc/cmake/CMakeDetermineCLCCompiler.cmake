include(${CMAKE_ROOT}/Modules/CMakeDetermineCompiler.cmake)

if(NOT CMAKE_CLC_COMPILER)
  find_program(CMAKE_CLC_COMPILER NAMES clang)
endif()
mark_as_advanced(CMAKE_CLC_COMPILER)

if(NOT CMAKE_CLC_ARCHIVE)
  find_program(CMAKE_CLC_ARCHIVE NAMES llvm-link)
endif()
mark_as_advanced(CMAKE_CLC_ARCHIVE)

set(CMAKE_CLC_COMPILER_ENV_VAR "CLC_COMPILER")
set(CMAKE_CLC_ARCHIVE_ENV_VAR "CLC_LINKER")
find_file(clc_comp_in CMakeCLCCompiler.cmake.in PATHS ${CMAKE_ROOT}/Modules ${CMAKE_MODULE_PATH})
# configure all variables set in this file
configure_file(${clc_comp_in} ${CMAKE_PLATFORM_INFO_DIR}/CMakeCLCCompiler.cmake @ONLY)
mark_as_advanced(clc_comp_in)
