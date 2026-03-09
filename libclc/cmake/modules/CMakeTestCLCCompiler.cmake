configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeCLCCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeCLCCompiler.cmake @ONLY)
include(${CMAKE_PLATFORM_INFO_DIR}/CMakeCLCCompiler.cmake)

include(AddLibclc)

if(CMAKE_CLC_COMPILER_FORCED)
  set(CMAKE_CLC_COMPILER_WORKS TRUE)
  return()
endif()

set(_test_file "${CMAKE_CURRENT_LIST_DIR}/CMakeCLCCompilerTest.cl")
set(_test_dir "${CMAKE_PLATFORM_INFO_DIR}/CMakeTmp")
set(_test_out "${_test_dir}/test_clc.o")
file(MAKE_DIRECTORY "${_test_dir}")

message(STATUS "Check for working CLC compiler: ${CMAKE_CLC_COMPILER}")

# Test that the compiler works for all targets in LIBCLC_TARGETS_TO_BUILD
foreach(_target ${LIBCLC_TARGETS_TO_BUILD})
  # Convert libclc target to clang triple
  libclc_target_to_clang_triple(${_target} _clang_triple)

  execute_process(
    COMMAND "${CMAKE_CLC_COMPILER}" --target=${_clang_triple} -x cl -c -flto
            -nostdlib -nostdlibinc -cl-no-stdinc -o "${_test_out}" "${_test_file}"
    RESULT_VARIABLE _clc_result
    ERROR_VARIABLE _clc_error
    OUTPUT_QUIET
  )

  if(NOT _clc_result EQUAL 0)
    message(FATAL_ERROR
      "The CLC compiler\n"
      "  ${CMAKE_CLC_COMPILER}\n"
      "is not able to compile a simple OpenCL test program for ${_target}.\n"
      "Output:\n${_clc_error}")
  endif()

  file(REMOVE "${_test_out}")
endforeach()

set(CMAKE_CLC_COMPILER_WORKS TRUE)
message(STATUS "Check for working CLC compiler: ${CMAKE_CLC_COMPILER} - works")
