# Converts a list of relative source paths to absolute paths and exports
# it to the parent scope.
macro(libclc_configure_source_list variable path)
  set(${variable} ${ARGN})
  list(TRANSFORM ${variable} PREPEND "${path}/")
  set(${variable} ${${variable}} PARENT_SCOPE)
endmacro()

# Merges OpenCL C source file lists with priority deduplication.
#
# All arguments after the output variable name are treated as source file
# paths. When multiple files share the same basename, the last occurrence
# wins. This allows target-specific files to automatically override generic
# ones.
function(libclc_merge_sources output)
  set(all_sources ${ARGN})
  set(result)
  set(seen_names)

  list(REVERSE all_sources)
  foreach(f ${all_sources})
    get_filename_component(name "${f}" NAME)
    if(NOT name IN_LIST seen_names)
      list(APPEND seen_names "${name}")
      list(PREPEND result "${f}")
    endif()
  endforeach()

  set(${output} ${result} PARENT_SCOPE)
endfunction()

# Creates a static library target for libclc builtins. Derives include
# directories to locate `.inc` files in the same directory.
function(add_libclc_builtin_library target_name)
  cmake_parse_arguments(ARG
    ""
    "FOLDER"
    "SOURCES;COMPILE_OPTIONS;INCLUDE_DIRS;COMPILE_DEFINITIONS"
    ${ARGN}
  )

  set(_inc_dirs)
  foreach(f ${ARG_SOURCES})
    get_filename_component(dir ${f} DIRECTORY)
    list(APPEND _inc_dirs ${dir})
  endforeach()
  list(REMOVE_DUPLICATES _inc_dirs)

  add_library(${target_name} STATIC ${ARG_SOURCES})
  target_compile_options(${target_name} PRIVATE ${ARG_COMPILE_OPTIONS})
  target_include_directories(${target_name} PRIVATE
    ${ARG_INCLUDE_DIRS} ${_inc_dirs}
  )
  target_compile_definitions(${target_name} PRIVATE ${ARG_COMPILE_DEFINITIONS})
  set_target_properties(${target_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    FOLDER ${ARG_FOLDER}
  )
endfunction()
