function(find_compiler_rt_library name variable)
  set(CLANG_COMMAND ${CMAKE_CXX_COMPILER} ${SANITIZER_COMMON_CFLAGS}
      "--rtlib=compiler-rt" "--print-libgcc-file-name")
  if (CMAKE_CXX_COMPILER_ID MATCHES Clang AND CMAKE_CXX_COMPILER_TARGET)
    list(APPEND CLANG_COMMAND "--target=${CMAKE_CXX_COMPILER_TARGET}")
  endif()
  get_property(SANITIZER_CXX_FLAGS CACHE CMAKE_CXX_FLAGS PROPERTY VALUE)
  string(REPLACE " " ";" SANITIZER_CXX_FLAGS "${SANITIZER_CXX_FLAGS}")
  list(APPEND CLANG_COMMAND ${SANITIZER_CXX_FLAGS})
  execute_process(
      COMMAND ${CLANG_COMMAND}
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE LIBRARY_FILE
  )
  string(STRIP "${LIBRARY_FILE}" LIBRARY_FILE)
  file(TO_CMAKE_PATH "${LIBRARY_FILE}" LIBRARY_FILE)
  string(REPLACE "builtins" "${name}" LIBRARY_FILE "${LIBRARY_FILE}")
  if (NOT HAD_ERROR AND EXISTS "${LIBRARY_FILE}")
    message(STATUS "Found compiler-rt ${name} library: ${LIBRARY_FILE}")
    set(${variable} "${LIBRARY_FILE}" PARENT_SCOPE)
  else()
    message(STATUS "Failed to find compiler-rt ${name} library")
  endif()
endfunction()
