# Check if compile-rt library file path exists.
# If found, cache the path in:
#    COMPILER_RT_LIBRARY-<name>-<target>
# If err_flag is true OR path not found, emit a message and set:
#    COMPILER_RT_LIBRARY-<name>-<target> to NOTFOUND
function(cache_compiler_rt_library err_flag name target library_file)
  if(err_flag OR NOT EXISTS "${library_file}")
    message(STATUS "Failed to find compiler-rt ${name} library for ${target}")
    set(COMPILER_RT_LIBRARY-${name}-${target} "NOTFOUND" CACHE INTERNAL
        "compiler-rt ${name} library for ${target}")
  else()
    message(STATUS "Found compiler-rt ${name} library: ${library_file}")
    set(COMPILER_RT_LIBRARY-${name}-${target} "${library_file}" CACHE INTERNAL
        "compiler-rt ${name} library for ${target}")
  endif()
endfunction()

# Find the path to compiler-rt library `name` (e.g. "builtins") for
# the specified `target` (e.g. "x86_64-linux") and return it in `variable`.
# This calls cache_compiler_rt_library that caches the path to speed up
# repeated invocations with the same `name` and `target`.
function(find_compiler_rt_library name target variable)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set(${variable} "NOTFOUND" PARENT_SCOPE)
    return()
  endif()
  if (NOT target AND CMAKE_CXX_COMPILER_TARGET)
    set(target "${CMAKE_CXX_COMPILER_TARGET}")
  endif()
  if(NOT DEFINED COMPILER_RT_LIBRARY-builtins-${target})
    # If the cache variable is not defined, invoke clang and then 
    # set it with cache_compiler_rt_library.
    set(CLANG_COMMAND ${CMAKE_CXX_COMPILER} ${SANITIZER_COMMON_FLAGS}
        "--rtlib=compiler-rt" "-print-libgcc-file-name")
    if(target)
      list(APPEND CLANG_COMMAND "--target=${target}")
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
    cache_compiler_rt_library(${HAD_ERROR}
      builtins "${target}" "${LIBRARY_FILE}")
  endif()
  if(NOT COMPILER_RT_LIBRARY-builtins-${target})
    set(${variable} "NOTFOUND" PARENT_SCOPE)
    return()
  endif()
  if(NOT DEFINED COMPILER_RT_LIBRARY-${name}-${target})
    # clang gives only the builtins library path. Other library paths are
    # obtained by substituting "builtins" with ${name} in the builtins
    # path and then checking if the resultant path exists. The result of
    # this check is also cached by cache_compiler_rt_library.
    set(LIBRARY_FILE "${COMPILER_RT_LIBRARY-builtins-${target}}")
    string(REPLACE "builtins" "${name}" LIBRARY_FILE "${LIBRARY_FILE}")
    cache_compiler_rt_library(FALSE "${name}" "${target}" "${LIBRARY_FILE}")
  endif()
  set(${variable} "${COMPILER_RT_LIBRARY-${name}-${target}}" PARENT_SCOPE)
endfunction()
