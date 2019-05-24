add_compiler_rt_component(crt)

function(check_cxx_section_exists section output)
  cmake_parse_arguments(ARG "" "" "SOURCE;FLAGS" ${ARGN})
  if(NOT ARG_SOURCE)
    set(ARG_SOURCE "int main() { return 0; }\n")
  endif()

  string(RANDOM TARGET_NAME)
  set(TARGET_NAME "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/cmTC_${TARGET_NAME}.dir")
  file(MAKE_DIRECTORY ${TARGET_NAME})

  file(WRITE "${TARGET_NAME}/CheckSectionExists.c" "${ARG_SOURCE}\n")

  string(REGEX MATCHALL "<[A-Za-z0-9_]*>" substitutions
         ${CMAKE_C_COMPILE_OBJECT})

  set(try_compile_flags "${ARG_FLAGS}")
  if(CMAKE_C_COMPILER_ID MATCHES Clang AND CMAKE_C_COMPILER_TARGET)
    list(APPEND try_compile_flags "-target ${CMAKE_C_COMPILER_TARGET}")
  endif()

  string(REPLACE ";" " " extra_flags "${try_compile_flags}")

  set(test_compile_command "${CMAKE_C_COMPILE_OBJECT}")
  foreach(substitution ${substitutions})
    if(substitution STREQUAL "<CMAKE_C_COMPILER>")
      string(REPLACE "<CMAKE_C_COMPILER>" "${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1}"
             test_compile_command ${test_compile_command})
    elseif(substitution STREQUAL "<OBJECT>")
      string(REPLACE "<OBJECT>" "${TARGET_NAME}/CheckSectionExists.o"
             test_compile_command ${test_compile_command})
    elseif(substitution STREQUAL "<SOURCE>")
      string(REPLACE "<SOURCE>" "${TARGET_NAME}/CheckSectionExists.c"
             test_compile_command ${test_compile_command})
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
    RESULT_VARIABLE TEST_RESULT
    OUTPUT_VARIABLE TEST_OUTPUT
    ERROR_VARIABLE TEST_ERROR
  )

  execute_process(
    COMMAND ${CMAKE_OBJDUMP} -h "${TARGET_NAME}/CheckSectionExists.o"
    RESULT_VARIABLE CHECK_RESULT
    OUTPUT_VARIABLE CHECK_OUTPUT
    ERROR_VARIABLE CHECK_ERROR
  )
  string(FIND "${CHECK_OUTPUT}" "${section}" SECTION_FOUND)

  if(NOT SECTION_FOUND EQUAL -1)
    set(${output} TRUE PARENT_SCOPE)
  else()
    set(${output} FALSE PARENT_SCOPE)
  endif()

  file(REMOVE_RECURSE ${TARGET_NAME})
endfunction()

check_cxx_section_exists(".init_array" COMPILER_RT_HAS_INITFINI_ARRAY
  SOURCE "__attribute__((constructor)) void f() {}\nint main() { return 0; }\n")

append_list_if(COMPILER_RT_HAS_STD_C11_FLAG -std=c11 CRT_CFLAGS)
append_list_if(COMPILER_RT_HAS_INITFINI_ARRAY -DCRT_HAS_INITFINI_ARRAY CRT_CFLAGS)
append_list_if(COMPILER_RT_HAS_FPIC_FLAG -fPIC CRT_CFLAGS)
append_list_if(COMPILER_RT_HAS_WNO_PEDANTIC -Wno-pedantic CRT_CFLAGS)

foreach(arch ${CRT_SUPPORTED_ARCH})
  add_compiler_rt_runtime(clang_rt.crtbegin
    OBJECT
    ARCHS ${arch}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/crtbegin.c
    CFLAGS ${CRT_CFLAGS}
    PARENT_TARGET crt)
  add_compiler_rt_runtime(clang_rt.crtend
    OBJECT
    ARCHS ${arch}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/crtend.c
    CFLAGS ${CRT_CFLAGS}
    PARENT_TARGET crt)
endforeach()
