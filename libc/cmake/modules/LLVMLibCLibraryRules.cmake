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
endfunction(collect_object_file_deps)

# A rule to build a library from a collection of entrypoint objects.
# Usage:
#     add_entrypoint_library(
#       DEPENDS <list of add_entrypoint_object targets>
#       EXT_DEPS <list of external object targets, no type checking is done>
#     )
#
# NOTE: If one wants an entrypoint to be availabe in a library, then they will
# have to list the entrypoint target explicitly in the DEPENDS list. Implicit
# entrypoint dependencies will not be added to the library.
function(add_entrypoint_library target_name)
  cmake_parse_arguments(
    "ENTRYPOINT_LIBRARY"
    "" # No optional arguments
    "" # No single value arguments
    "DEPENDS;EXT_DEPS" # Multi-value arguments
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
    if(NOT (${dep_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE}))
      message(FATAL_ERROR "Dependency '${dep}' of 'add_entrypoint_collection' is "
                          "not an 'add_entrypoint_object' target.")
    endif()
    collect_object_file_deps(${dep} recursive_deps)
    list(APPEND all_deps ${recursive_deps})
  endforeach(dep)
  list(REMOVE_DUPLICATES all_deps)
  set(objects "")
  foreach(dep IN LISTS all_deps)
    list(APPEND objects $<TARGET_OBJECTS:${dep}>)
  endforeach(dep)

  foreach(dep IN LISTS ENTRYPOINT_LIBRARY_EXT_DEPS)
    list(APPEND objects $<TARGET_OBJECTS:${dep}>)
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

# Rule to add header only libraries.
# Usage
#    add_header_library(
#      <target name>
#      HDRS  <list of .h files part of the library>
#      DEPENDS <list of dependencies>
#    )
function(add_header_library target_name)
  cmake_parse_arguments(
    "ADD_HEADER"
    "" # No optional arguments
    "" # No Single value arguments
    "HDRS;DEPENDS" # Multi-value arguments
    ${ARGN}
  )

  if(NOT ADD_HEADER_HDRS)
    message(FATAL_ERROR "'add_header_library' target requires a HDRS list of .h files.")
  endif()

  get_fq_target_name(${target_name} fq_target_name)

  set(FULL_HDR_PATHS "")
  # TODO: Remove this foreach block when we can switch to the new
  # version of the CMake policy CMP0076.
  foreach(hdr IN LISTS ADD_HEADER_HDRS)
    list(APPEND FULL_HDR_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/${hdr})
  endforeach()

  set(interface_target_name "${fq_target_name}.__header_library__")

  add_library(${interface_target_name} INTERFACE)
  target_sources(${interface_target_name} INTERFACE ${FULL_HDR_PATHS})
  get_fq_deps_list(fq_deps_list ${ADD_HEADER_DEPENDS})
  if(ADD_HEADER_DEPENDS)
    add_dependencies(${interface_target_name} ${fq_deps_list})
  endif()

  add_custom_target(${fq_target_name})
  add_dependencies(${fq_target_name} ${interface_target_name})
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      "TARGET_TYPE" "${HDR_LIBRARY_TARGET_TYPE}"
      "DEPS" "${fq_deps_list}"
  )
endfunction(add_header_library)
