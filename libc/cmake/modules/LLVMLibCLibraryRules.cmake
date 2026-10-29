function(collect_object_file_deps target result)
  # NOTE: This function does add entrypoint targets to |result|.
  # It is expected that the caller adds them separately.
  #
  # Header libraries are dense in the dependency graph, so each target keeps
  # the result of its walk and every later visit is a lookup. A target reached
  # again while its own walk is still in progress adds nothing, which ends a
  # cycle. A dependency named but never defined, as some header libraries
  # have, has nothing to add either.
  if(NOT TARGET ${target})
    set(${result} "" PARENT_SCOPE)
    return()
  endif()
  get_target_property(aliased ${target} "ALIASED_TARGET")
  if(aliased)
    set(target ${aliased})
  endif()
  get_target_property(done ${target} "COLLECTED_OBJECT_DEPS_DONE")
  if(done)
    get_target_property(all_deps ${target} "COLLECTED_OBJECT_DEPS")
    set(${result} ${all_deps} PARENT_SCOPE)
    return()
  endif()
  get_target_property(in_progress ${target} "COLLECTING_OBJECT_DEPS")
  get_target_property(target_type ${target} "TARGET_TYPE")
  if(in_progress OR NOT target_type)
    set(${result} "" PARENT_SCOPE)
    return()
  endif()
  set_target_properties(${target} PROPERTIES COLLECTING_OBJECT_DEPS TRUE)

  set(all_deps "")
  get_target_property(deps ${target} "DEPS")
  if(${target_type} STREQUAL ${ENTRYPOINT_EXT_TARGET_TYPE})
    # It is not possible to recursively extract deps of external dependencies.
    # So, we just accumulate the direct dep and return.
    set(all_deps ${deps})
  elseif((${target_type} STREQUAL ${OBJECT_LIBRARY_TARGET_TYPE}) OR
      (${target_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE}) OR
      (${target_type} STREQUAL ${HDR_LIBRARY_TARGET_TYPE}))
    if(${target_type} STREQUAL ${OBJECT_LIBRARY_TARGET_TYPE})
      list(APPEND all_deps ${target})
    endif()
    foreach(dep IN LISTS deps)
      collect_object_file_deps(${dep} dep_targets)
      list(APPEND all_deps ${dep_targets})
    endforeach()
    list(REMOVE_DUPLICATES all_deps)
  endif()

  set_target_properties(${target} PROPERTIES
    COLLECTED_OBJECT_DEPS "${all_deps}"
    COLLECTED_OBJECT_DEPS_DONE TRUE)
  set(${result} ${all_deps} PARENT_SCOPE)
endfunction()

function(get_all_object_file_deps result fq_deps_list)
  set(all_deps "")
  foreach(dep ${fq_deps_list})
    get_target_property(dep_type ${dep} "TARGET_TYPE")
    if(NOT ((${dep_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE}) OR
      (${dep_type} STREQUAL ${ENTRYPOINT_EXT_TARGET_TYPE})))
      message(FATAL_ERROR "Dependency '${dep}' of 'add_entrypoint_collection' is "
        "not an 'add_entrypoint_object' or 'add_entrypoint_external' target.")
    endif()
    collect_object_file_deps(${dep} recursive_deps)
    list(APPEND all_deps ${recursive_deps})
    # Add the entrypoint object target explicitly as collect_object_file_deps
    # only collects object files from non-entrypoint targets.
    if(${dep_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE})
      set(entrypoint_target ${dep})
      get_target_property(is_alias ${entrypoint_target} "IS_ALIAS")
      if(is_alias)
        get_target_property(aliasee ${entrypoint_target} "DEPS")
        if(NOT aliasee)
          message(FATAL_ERROR
            "Entrypoint alias ${entrypoint_target} does not have an aliasee.")
        endif()
        set(entrypoint_target ${aliasee})
      endif()
    endif()
    list(APPEND all_deps ${entrypoint_target})
  endforeach()
  list(REMOVE_DUPLICATES all_deps)
  set(${result} ${all_deps} PARENT_SCOPE)
endfunction()

# Link bitcode inputs into a single LLVM IR module using llvm-link. Build
# static archive library firstly and use this static archive library as
# as llvm-link's input.
# Usage:
#     llvm_link_bitcode(
#       target_name
#       OUTPUT <output file path>
#       INPUTS <input file path>
#       DEPENDS <targets that must be built first>
#     )
function(llvm_link_bitcode target_name)
  cmake_parse_arguments(
    "ARG"
    ""
    "OUTPUT"
    "INPUTS;DEPENDS"
    ${ARGN}
  )
  add_custom_command(
    OUTPUT ${ARG_OUTPUT}
    COMMAND ${LIBC_LLVM_LINK} ${ARG_INPUTS} -o ${ARG_OUTPUT}
    DEPENDS ${ARG_DEPENDS}
    COMMENT "Linking bitcode ${ARG_OUTPUT}"
  )
  add_custom_target(${target_name} ALL DEPENDS ${ARG_OUTPUT})
  set_target_properties(${target_name} PROPERTIES TARGET_FILE ${ARG_OUTPUT})
endfunction()

# A rule to build a library from a collection of entrypoint objects and bundle
# it in a single LLVM IR module.
# Usage:
#     add_bitcode_entrypoint_library(
#       target_name
#       base_target_name
#       OUTPUT_NAME <output file name>
#     )
function(add_bitcode_entrypoint_library target_name base_target_name)
  cmake_parse_arguments(
    "ARG"
    ""
    "OUTPUT_NAME"
    ""
    ${ARGN}
  )
  if(NOT ARG_OUTPUT_NAME)
    message(FATAL_ERROR "'add_bitcode_entrypoint_library' target requires an "
      "OUTPUT_NAME.")
  endif()
  llvm_link_bitcode(${target_name}
    OUTPUT ${LIBC_LIBRARY_DIR}/${ARG_OUTPUT_NAME}
    INPUTS $<TARGET_FILE:${base_target_name}>
    DEPENDS ${base_target_name}
  )
endfunction()

# A rule to build a library from a collection of entrypoint objects.
# Usage:
#     add_entrypoint_library(
#       target_name
#       [SHARED]
#       [SOVERSION <version>]
#       DEPENDS <list of add_entrypoint_object targets>
#     )
#
# Without SHARED a static archive is produced. SHARED builds a shared object
# instead and requires LIBC_ENABLE_SHARED, which makes the entrypoint objects
# position independent.
#
# NOTE: If one wants an entrypoint to be available in a library, then they will
# have to list the entrypoint target explicitly in the DEPENDS list. Implicit
# entrypoint dependencies will not be added to the library.
function(add_entrypoint_library target_name)
  cmake_parse_arguments(
    "ENTRYPOINT_LIBRARY"
    "SHARED" # Optional arguments
    "SOVERSION" # Single value arguments
    "DEPENDS" # Multi-value arguments
    ${ARGN}
  )
  if(NOT ENTRYPOINT_LIBRARY_DEPENDS)
    message(FATAL_ERROR "'add_entrypoint_library' target requires a DEPENDS list "
      "of 'add_entrypoint_object' targets.")
  endif()
  if(ENTRYPOINT_LIBRARY_SHARED AND NOT LIBC_ENABLE_SHARED)
    message(FATAL_ERROR
      "'add_entrypoint_library' with SHARED requires LIBC_ENABLE_SHARED.")
  endif()

  get_fq_deps_list(fq_deps_list ${ENTRYPOINT_LIBRARY_DEPENDS})
  get_all_object_file_deps(all_deps "${fq_deps_list}")

  set(objects "")
  foreach(dep IN LISTS all_deps)
    list(APPEND objects $<$<STREQUAL:$<TARGET_NAME_IF_EXISTS:${dep}>,${dep}>:$<TARGET_OBJECTS:${dep}>>)
  endforeach()

  if(NOT ENTRYPOINT_LIBRARY_SHARED)
    add_library(
      ${target_name}
      STATIC
        ${objects}
    )
    set_target_properties(${target_name} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${LIBC_LIBRARY_DIR})
    return()
  endif()

  add_library(
    ${target_name}
    SHARED
      ${objects}
  )
  set_target_properties(${target_name}
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIBC_LIBRARY_DIR})
  if(NOT ENTRYPOINT_LIBRARY_SOVERSION STREQUAL "")
    set_target_properties(${target_name}
      PROPERTIES SOVERSION ${ENTRYPOINT_LIBRARY_SOVERSION})
  endif()
  # Link nothing from the host toolchain except the compiler runtime, which the
  # math and fixed-point entrypoints need for builtins such as __divti3. The
  # driver defaults would otherwise add a DT_NEEDED on the system libm.
  target_link_options(${target_name} PRIVATE -nostdlib)
  if(LIBC_COMPILER_RT_BUILTINS)
    target_link_libraries(${target_name} PRIVATE ${LIBC_COMPILER_RT_BUILTINS})
  endif()
endfunction()

set(HDR_LIBRARY_TARGET_TYPE "HDR_LIBRARY")

# Internal function, used by `add_header_library`.
function(create_header_library fq_target_name)
  cmake_parse_arguments(
    "ADD_HEADER"
    "" # Optional arguments
    "" # Single value arguments
    "HDRS;DEPENDS;FLAGS;COMPILE_OPTIONS" # Multi-value arguments
    ${ARGN}
  )

  if(NOT ADD_HEADER_HDRS)
    message(FATAL_ERROR "'add_header_library' target requires a HDRS list of .h files.")
  endif()

  if(SHOW_INTERMEDIATE_OBJECTS)
    message(STATUS "Adding header library ${fq_target_name}")
    if(${SHOW_INTERMEDIATE_OBJECTS} STREQUAL "DEPS")
      foreach(dep IN LISTS ADD_HEADER_DEPENDS)
        message(STATUS "  ${fq_target_name} depends on ${dep}")
      endforeach()
    endif()
  endif()

  add_library(${fq_target_name} INTERFACE)
  target_sources(${fq_target_name} INTERFACE ${ADD_HEADER_HDRS})
  if(ADD_HEADER_DEPENDS)
    add_dependencies(${fq_target_name} ${ADD_HEADER_DEPENDS})

    # `*.__copied_hdr__` is created only to copy the header files to the target
    # location, not to be linked against.
    set(link_lib "")
    foreach(dep ${ADD_HEADER_DEPENDS})
      if(NOT dep MATCHES "__copied_hdr__")
        list(APPEND link_lib ${dep})
      endif()
    endforeach()

    target_link_libraries(${fq_target_name} INTERFACE ${link_lib})
  endif()
  if(ADD_HEADER_COMPILE_OPTIONS)
    target_compile_options(${fq_target_name} INTERFACE ${ADD_HEADER_COMPILE_OPTIONS})
  endif()
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      INTERFACE_FLAGS "${ADD_HEADER_FLAGS}"
      TARGET_TYPE "${HDR_LIBRARY_TARGET_TYPE}"
      DEPS "${ADD_HEADER_DEPENDS}"
      FLAGS "${ADD_HEADER_FLAGS}"
  )
endfunction()

# Rule to add header only libraries.
# Usage
#    add_header_library(
#      <target name>
#      HDRS  <list of .h files part of the library>
#      DEPENDS <list of dependencies>
#      FLAGS <list of flags>
#    )

function(add_header_library target_name)
  add_target_with_flags(
    ${target_name}
    CREATE_TARGET create_header_library
    ${ARGN}
  )
endfunction()
