function(collect_object_file_deps target result)
  set(all_deps "")
  get_target_property(target_type ${target} "TARGET_TYPE")
  if(NOT target_type)
    return()
  endif()

  if(${target_type} STREQUAL ${OBJECT_LIBRARY_TARGET_TYPE})
    list(APPEND all_deps ${target})
    get_target_property(deps ${target} "DEPS")
    foreach(dep IN LISTS deps)
      collect_object_file_deps(${dep} dep_targets)
      list(APPEND all_deps ${dep_targets})
    endforeach(dep)
    set(${result} ${all_deps} PARENT_SCOPE)
    return()
  endif()

  if(${target_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE})
    set(entrypoint_target ${target})
    get_target_property(is_alias ${entrypoint_target} "IS_ALIAS")
    if(is_alias)
      get_target_property(aliasee ${entrypoint_target} "DEPS")
      if(NOT aliasee)
        message(FATAL_ERROR
                "Entrypoint alias ${entrypoint_target} does not have an aliasee.")
      endif()
      set(entrypoint_target ${aliasee})
    endif()
    list(APPEND all_deps ${entrypoint_target})
    get_target_property(deps ${target} "DEPS")
    foreach(dep IN LISTS deps)
      collect_object_file_deps(${dep} dep_targets)
      list(APPEND all_deps ${dep_targets})
    endforeach(dep)
    set(${result} ${all_deps} PARENT_SCOPE)
    return()
  endif()

  if(${target_type} STREQUAL ${ENTRYPOINT_EXT_TARGET_TYPE})
    # It is not possible to recursively extract deps of external dependencies.
    # So, we just accumulate the direct dep and return.
    get_target_property(deps ${target} "DEPS")
    set(${result} ${deps} PARENT_SCOPE)
    return()
  endif()
endfunction(collect_object_file_deps)

# A rule to build a library from a collection of entrypoint objects.
# Usage:
#     add_entrypoint_library(
#       DEPENDS <list of add_entrypoint_object targets>
#     )
#
# NOTE: If one wants an entrypoint to be available in a library, then they will
# have to list the entrypoint target explicitly in the DEPENDS list. Implicit
# entrypoint dependencies will not be added to the library.
function(add_entrypoint_library target_name)
  cmake_parse_arguments(
    "ENTRYPOINT_LIBRARY"
    "" # No optional arguments
    "" # No single value arguments
    "DEPENDS" # Multi-value arguments
    ${ARGN}
  )
  if(NOT ENTRYPOINT_LIBRARY_DEPENDS)
    message(FATAL_ERROR "'add_entrypoint_library' target requires a DEPENDS list "
                        "of 'add_entrypoint_object' targets.")
  endif()

  get_fq_deps_list(fq_deps_list ${ENTRYPOINT_LIBRARY_DEPENDS})
  set(all_deps "")
  foreach(dep IN LISTS fq_deps_list)
    get_target_property(dep_type ${dep} "TARGET_TYPE")
    if(NOT ((${dep_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE}) OR (${dep_type} STREQUAL ${ENTRYPOINT_EXT_TARGET_TYPE})))
      message(FATAL_ERROR "Dependency '${dep}' of 'add_entrypoint_collection' is "
                          "not an 'add_entrypoint_object' or 'add_entrypoint_external' target.")
    endif()
    collect_object_file_deps(${dep} recursive_deps)
    list(APPEND all_deps ${recursive_deps})
  endforeach(dep)
  list(REMOVE_DUPLICATES all_deps)
  set(objects "")
  foreach(dep IN LISTS all_deps)
    list(APPEND objects $<$<STREQUAL:$<TARGET_NAME_IF_EXISTS:${dep}>,${dep}>:$<TARGET_OBJECTS:${dep}>>)
  endforeach(dep)

  add_library(
    ${target_name}
    STATIC
    ${objects}
  )
  set_target_properties(${target_name}  PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction(add_entrypoint_library)

# Rule to build a shared library of redirector objects.
function(add_redirector_library target_name)
  cmake_parse_arguments(
    "REDIRECTOR_LIBRARY"
    ""
    ""
    "DEPENDS"
    ${ARGN}
  )

  set(obj_files "")
  foreach(dep IN LISTS REDIRECTOR_LIBRARY_DEPENDS)
    # TODO: Ensure that each dep is actually a add_redirector_object target.
    list(APPEND obj_files $<TARGET_OBJECTS:${dep}>)
  endforeach(dep)

  # TODO: Call the linker explicitly instead of calling the compiler driver to
  # prevent DT_NEEDED on C++ runtime.
  add_library(
    ${target_name}
    EXCLUDE_FROM_ALL
    SHARED
    ${obj_files}
  )
  set_target_properties(${target_name}  PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(${target_name}  -nostdlib -lc -lm)
  set_target_properties(${target_name}  PROPERTIES LINKER_LANGUAGE "C")
endfunction(add_redirector_library)

set(HDR_LIBRARY_TARGET_TYPE "HDR_LIBRARY")

# Internal function, used by `add_header_library`.
function(create_header_library fq_target_name)
  cmake_parse_arguments(
    "ADD_HEADER"
    "" # Optional arguments
    "" # Single value arguments
    "HDRS;DEPENDS;FLAGS" # Multi-value arguments
    ${ARGN}
  )

  if(NOT ADD_HEADER_HDRS)
    message(FATAL_ERROR "'add_header_library' target requires a HDRS list of .h files.")
  endif()

  set(FULL_HDR_PATHS "")
  # TODO: Remove this foreach block when we can switch to the new
  # version of the CMake policy CMP0076.
  foreach(hdr IN LISTS ADD_HEADER_HDRS)
    list(APPEND FULL_HDR_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/${hdr})
  endforeach()

  if(SHOW_INTERMEDIATE_OBJECTS)
    message(STATUS "Adding header library ${fq_target_name}")
    if(${SHOW_INTERMEDIATE_OBJECTS} STREQUAL "DEPS")
      foreach(dep IN LISTS ADD_HEADER_DEPENDS)
        message(STATUS "  ${fq_target_name} depends on ${dep}")
      endforeach()
    endif()
  endif()
  set(interface_target_name "${fq_target_name}.__header_library__")

  add_library(${interface_target_name} INTERFACE)
  target_sources(${interface_target_name} INTERFACE ${FULL_HDR_PATHS})
  if(ADD_HEADER_DEPENDS)
    add_dependencies(${interface_target_name} ${ADD_HEADER_DEPENDS})
  endif()
  set_target_properties(
    ${interface_target_name}
    PROPERTIES
      INTERFACE_FLAGS "${ADD_HEADER_FLAGS}"
  )

  add_custom_target(${fq_target_name})
  add_dependencies(${fq_target_name} ${interface_target_name})
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      TARGET_TYPE "${HDR_LIBRARY_TARGET_TYPE}"
      DEPS "${ADD_HEADER_DEPENDS}"
      FLAGS "${ADD_HEADER_FLAGS}"
  )
endfunction(create_header_library)

# Rule to add header only libraries.
# Usage
#    add_header_library(
#      <target name>
#      HDRS  <list of .h files part of the library>
#      DEPENDS <list of dependencies>
#      FLAGS <list of flags>
#    )

# Internal function, used by `add_header_library`.
function(expand_flags_for_header_library target_name flags)
  cmake_parse_arguments(
    "EXPAND_FLAGS"
    "IGNORE_MARKER" # Optional arguments
    "" # Single-value arguments
    "DEPENDS;FLAGS" # Multi-value arguments
    ${ARGN}
  )

  list(LENGTH flags nflags)
  if(NOT ${nflags})
    create_header_library(
      ${target_name}
      DEPENDS ${EXPAND_FLAGS_DEPENDS}
      FLAGS ${EXPAND_FLAGS_FLAGS}
      ${EXPAND_FLAGS_UNPARSED_ARGUMENTS}
    )
    return()
  endif()

  list(GET flags 0 flag)
  list(REMOVE_AT flags 0)
  extract_flag_modifier(${flag} real_flag modifier)

  if(NOT "${modifier}" STREQUAL "NO")
    expand_flags_for_header_library(
      ${target_name}
      "${flags}"
      DEPENDS ${EXPAND_FLAGS_DEPENDS} IGNORE_MARKER
      FLAGS ${EXPAND_FLAGS_FLAGS} IGNORE_MARKER
      ${EXPAND_FLAGS_UNPARSED_ARGUMENTS}
    )
  endif()

  if("${real_flag}" STREQUAL "" OR "${modifier}" STREQUAL "ONLY")
    return()
  endif()

  set(NEW_FLAGS ${EXPAND_FLAGS_FLAGS})
  list(REMOVE_ITEM NEW_FLAGS ${flag})
  get_fq_dep_list_without_flag(NEW_DEPS ${real_flag} ${EXPAND_FLAGS_DEPENDS})

  # Only target with `flag` has `.__NO_flag` target, `flag__NO` and
  # `flag__ONLY` do not.
  if(NOT "${modifier}")
    set(TARGET_NAME "${target_name}.__NO_${flag}")
  else()
    set(TARGET_NAME "${target_name}")
  endif()

  expand_flags_for_header_library(
    ${TARGET_NAME}
    "${flags}"
    DEPENDS ${NEW_DEPS} IGNORE_MARKER
    FLAGS ${NEW_FLAGS} IGNORE_MARKER
    ${EXPAND_FLAGS_UNPARSED_ARGUMENTS}
  )
endfunction(expand_flags_for_header_library)

function(add_header_library target_name)
  cmake_parse_arguments(
    "ADD_TO_EXPAND"
    "" # Optional arguments
    "" # Single value arguments
    "DEPENDS;FLAGS" # Multi-value arguments
    ${ARGN}
  )

  get_fq_target_name(${target_name} fq_target_name)

  if(ADD_TO_EXPAND_DEPENDS AND ("${SHOW_INTERMEDIATE_OBJECTS}" STREQUAL "DEPS"))
    message(STATUS "Gathering FLAGS from dependencies for ${fq_target_name}")
  endif()

  get_fq_deps_list(fq_deps_list ${ADD_TO_EXPAND_DEPENDS})
  get_flags_from_dep_list(deps_flag_list ${fq_deps_list})
  
  list(APPEND ADD_TO_EXPAND_FLAGS ${deps_flag_list})
  remove_duplicated_flags("${ADD_TO_EXPAND_FLAGS}" flags)
  list(SORT flags)

  if(SHOW_INTERMEDIATE_OBJECTS AND flags)
    message(STATUS "Header library ${fq_target_name} has FLAGS: ${flags}")
  endif()

  expand_flags_for_header_library(
    ${fq_target_name}
    "${flags}"
    DEPENDS ${fq_deps_list} IGNORE_MARKER
    FLAGS ${flags} IGNORE_MARKER
    ${ADD_TO_EXPAND_UNPARSED_ARGUMENTS}
  )
endfunction(add_header_library)
