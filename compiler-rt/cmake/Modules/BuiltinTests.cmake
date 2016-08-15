
# This function takes an OS and a list of architectures and identifies the
# subset of the architectures list that the installed toolchain can target.
function(try_compile_only output)
  cmake_parse_arguments(ARG "" "" "SOURCE;FLAGS" ${ARGN})
  if(NOT ARG_SOURCE)
    set(ARG_SOURCE "int foo(int x, int y) { return x + y; }\n")
  endif()
  set(SIMPLE_C ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/src.c)
  file(WRITE ${SIMPLE_C} "${ARG_SOURCE}\n")
  string(REGEX MATCHALL "<[A-Za-z0-9_]*>" substitutions
         ${CMAKE_C_COMPILE_OBJECT})
  string(REPLACE ";" " " extra_flags "${ARG_FLAGS}")

  set(test_compile_command "${CMAKE_C_COMPILE_OBJECT}")
  foreach(substitution ${substitutions})
    if(substitution STREQUAL "<CMAKE_C_COMPILER>")
      string(REPLACE "<CMAKE_C_COMPILER>"
             "${CMAKE_C_COMPILER}" test_compile_command ${test_compile_command})
    elseif(substitution STREQUAL "<OBJECT>")
      string(REPLACE "<OBJECT>"
             "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/test.o"
             test_compile_command ${test_compile_command})
    elseif(substitution STREQUAL "<SOURCE>")
      string(REPLACE "<SOURCE>" "${SIMPLE_C}" test_compile_command
             ${test_compile_command})
    elseif(substitution STREQUAL "<FLAGS>")
      string(REPLACE "<FLAGS>" "${CMAKE_C_FLAGS} ${extra_flags}"
             test_compile_command ${test_compile_command})
    else()
      string(REPLACE "${substitution}" "" test_compile_command
             ${test_compile_command})
    endif()
  endforeach()

  string(REPLACE " " ";" test_compile_command "${test_compile_command}")

  execute_process(
    COMMAND ${test_compile_command}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE TEST_OUTPUT
    ERROR_VARIABLE TEST_ERROR
  )
  if(result EQUAL 0)
    set(${output} True PARENT_SCOPE)
  else()
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
        "Testing compiler for supporting " ${ARGN} ":\n"
        "Command: ${test_compile_command}\n"
        "${TEST_OUTPUT}\n${TEST_ERROR}\n${result}\n")
    set(${output} False PARENT_SCOPE)
  endif()
endfunction()

function(builtin_check_c_compiler_flag flag output)
  if(NOT DEFINED ${output})
    message(STATUS "Performing Test ${output}")
    try_compile_only(result FLAGS ${flag})
    set(${output} ${result} CACHE INTERNAL "Compiler supports ${flag}")
    if(${result})
      message(STATUS "Performing Test ${output} - Success")
    else()
      message(STATUS "Performing Test ${output} - Failed")
    endif()
  endif()
endfunction()

function(builtin_check_c_compiler_source output source)
  if(NOT DEFINED ${output})
    message(STATUS "Performing Test ${output}")
    try_compile_only(result SOURCE ${source})
    set(${output} ${result} CACHE INTERNAL "Compiler supports ${flag}")
    if(${result})
      message(STATUS "Performing Test ${output} - Success")
    else()
      message(STATUS "Performing Test ${output} - Failed")
    endif()
  endif()
endfunction()
