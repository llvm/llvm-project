# This is a helper function and not a build rule. It is to be used by the
# the "add_entrypoint_library" rule to generate the full list of object files
# recursively produced by "add_object_library" targets upstream in the
# dependency tree. This function traverses up through the
# "add_entrypoint_object" targets but does not collect the object files
# produced by them.
# Usage:
#   get_object_files_for_test(<result var> <target0> [<target1> ...])
#
#   targetN is either an "add_entrypoint_target" target or an
#   "add_object_library" target.
function(get_object_files_for_entrypoint_library result)
  set(object_files "")
  foreach(dep IN LISTS ARGN)
    get_target_property(dep_type ${dep} "TARGET_TYPE")
    if (NOT dep_type)
      continue()
    endif()

    if(${dep_type} STREQUAL ${OBJECT_LIBRARY_TARGET_TYPE})
      get_target_property(dep_object_files ${dep} "OBJECT_FILES")
      if(dep_object_files)
        list(APPEND object_files ${dep_object_files})
      endif()
    endif()

    get_target_property(indirect_deps ${dep} "DEPS")
    get_object_files_for_entrypoint_library(indirect_objfiles ${indirect_deps})
    list(APPEND object_files ${indirect_objfiles})
  endforeach(dep)
  list(REMOVE_DUPLICATES object_files)
  set(${result} ${object_files} PARENT_SCOPE)
endfunction()

# This is a helper function and not a build rule. Given an entrypoint object
# target, it returns the object file produced by this target in |result|.
# If the given entrypoint target is an alias, then it traverses up to the
# aliasee to get the object file.
function(get_entrypoint_object_file entrypoint_target result)
  get_target_property(target_type ${entrypoint_target} "TARGET_TYPE")
  if(NOT (${target_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE}))
    message(FATAL_ERROR
            "Expected an target added using `add_entrypoint_object` rule.")
  endif()

  get_target_property(objfile ${entrypoint_target} "OBJECT_FILE")
  if(objfile)
    set(${result} ${objfile} PARENT_SCOPE)
    return()
  endif()

  # If the entrypoint is an alias, fetch the object file from the aliasee.
  get_target_property(is_alias ${entrypoint_target} "IS_ALIAS")
  if(is_alias)
    get_target_property(aliasee ${entrypoint_target} "DEPS")
    if(NOT aliasee)
      message(FATAL_ERROR
              "Entrypoint alias ${entrypoint_target} does not have an aliasee.")
    endif()
    get_entrypoint_object_file(${aliasee} objfile)
    set(${result} ${objfile} PARENT_SCOPE)
    return()
  endif()

  message(FATAL_ERROR
          "Entrypoint ${entrypoint_target} does not produce an object file.")
endfunction(get_entrypoint_object_file)

# A rule to build a library from a collection of entrypoint objects.
# Usage:
#     add_entrypoint_library(
#       DEPENDS <list of add_entrypoint_object targets>
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
    "DEPENDS" # Multi-value arguments
    ${ARGN}
  )
  if(NOT ENTRYPOINT_LIBRARY_DEPENDS)
    message(FATAL_ERROR "'add_entrypoint_library' target requires a DEPENDS list "
                        "of 'add_entrypoint_object' targets.")
  endif()

  get_fq_deps_list(fq_deps_list ${ENTRYPOINT_LIBRARY_DEPENDS})
  get_object_files_for_entrypoint_library(obj_list ${fq_deps_list})
  foreach(dep IN LISTS fq_deps_list)
    get_target_property(dep_type ${dep} "TARGET_TYPE")
    if(NOT (${dep_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE}))
      message(FATAL_ERROR "Dependency '${dep}' of 'add_entrypoint_collection' is "
                          "not an 'add_entrypoint_object' target.")
    endif()
    get_entrypoint_object_file(${dep} objfile)
    list(APPEND obj_list ${objfile})
  endforeach(dep)
  list(REMOVE_DUPLICATES obj_list)

  set(library_file "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${target_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  add_custom_command(
    OUTPUT ${library_file}
    COMMAND ${CMAKE_AR} -r ${library_file} ${obj_list}
    DEPENDS ${obj_list}
  )
  add_custom_target(
    ${target_name}
    ALL
    DEPENDS ${library_file}
  )
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
  set_target_properties(${target_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  target_link_libraries(
    ${target_name}
    -nostdlib -lc -lm
  )

  set_target_properties(
    ${target_name}
    PROPERTIES
      LINKER_LANGUAGE "C"
  )
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
