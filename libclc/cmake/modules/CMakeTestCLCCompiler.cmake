configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeCLCCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeCLCCompiler.cmake @ONLY)
include(${CMAKE_PLATFORM_INFO_DIR}/CMakeCLCCompiler.cmake)

if(CMAKE_CLC_COMPILER_FORCED)
  set(CMAKE_CLC_COMPILER_WORKS TRUE)
  return()
endif()

set(_test_file "${CMAKE_CURRENT_LIST_DIR}/CMakeCLCCompilerTest.cl")
set(_test_dir "${CMAKE_PLATFORM_INFO_DIR}/CMakeTmp")
set(_test_out "${_test_dir}/test_clc.o")
file(MAKE_DIRECTORY "${_test_dir}")

message(STATUS "Check for working CLC compiler: ${CMAKE_CLC_COMPILER}")

execute_process(
  COMMAND "${CMAKE_CLC_COMPILER}" --target=spirv64-unknown-unknown -x cl -c -flto
          -disable-llvm-passes -o "${_test_out}" "${_test_file}"
  RESULT_VARIABLE _clc_result
  ERROR_VARIABLE _clc_error
)

if(_clc_result EQUAL 0)
  set(CMAKE_CLC_COMPILER_WORKS TRUE)
  message(STATUS "Check for working CLC compiler: ${CMAKE_CLC_COMPILER} - works")
  file(REMOVE "${_test_out}")
else()
  message(FATAL_ERROR
    "The CLC compiler\n"
    "  ${CMAKE_CLC_COMPILER}\n"
    "is not able to compile a simple OpenCL test program.\n"
    "Output:\n${_clc_error}")
endif()
