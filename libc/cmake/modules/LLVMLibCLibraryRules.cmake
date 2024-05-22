function(collect_object_file_deps target result)
  # NOTE: This function does add entrypoint targets to |result|.
  # It is expected that the caller adds them separately.
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
    list(REMOVE_DUPLICATES all_deps)
    set(${result} ${all_deps} PARENT_SCOPE)
    return()
  endif()

  if(${target_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE} OR
     ${target_type} STREQUAL ${ENTRYPOINT_OBJ_VENDOR_TARGET_TYPE})
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
    get_target_property(deps ${target} "DEPS")
    foreach(dep IN LISTS deps)
      collect_object_file_deps(${dep} dep_targets)
      list(APPEND all_deps ${dep_targets})
    endforeach(dep)
    list(REMOVE_DUPLICATES all_deps)
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

function(get_all_object_file_deps result fq_deps_list)
  set(all_deps "")
  foreach(dep ${fq_deps_list})
    get_target_property(dep_type ${dep} "TARGET_TYPE")
    if(NOT ((${dep_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE}) OR
            (${dep_type} STREQUAL ${ENTRYPOINT_EXT_TARGET_TYPE}) OR
            (${dep_type} STREQUAL ${ENTRYPOINT_OBJ_VENDOR_TARGET_TYPE})))
      message(FATAL_ERROR "Dependency '${dep}' of 'add_entrypoint_collection' is "
                          "not an 'add_entrypoint_object' or 'add_entrypoint_external' target.")
    endif()
    collect_object_file_deps(${dep} recursive_deps)
    list(APPEND all_deps ${recursive_deps})
    # Add the entrypoint object target explicitly as collect_object_file_deps
    # only collects object files from non-entrypoint targets.
    if(${dep_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE} OR
       ${dep_type} STREQUAL ${ENTRYPOINT_OBJ_VENDOR_TARGET_TYPE})
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
  endforeach(dep)
  list(REMOVE_DUPLICATES all_deps)
  set(${result} ${all_deps} PARENT_SCOPE)
endfunction()

# A rule to build a library from a collection of entrypoint objects and bundle
# it into a GPU fatbinary. Usage is the same as 'add_entrypoint_library'.
# Usage:
#     add_gpu_entrypoint_library(
#       DEPENDS <list of add_entrypoint_object targets>
#     )
function(add_gpu_entrypoint_library target_name base_target_name)
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
  get_all_object_file_deps(all_deps "${fq_deps_list}")

  # The GPU 'libc' needs to be exported in a format that can be linked with
  # offloading langauges like OpenMP or CUDA. This wraps every GPU object into a
  # fat binary and adds them to a static library.
  set(objects "")
  foreach(dep IN LISTS all_deps)
    set(object $<$<STREQUAL:$<TARGET_NAME_IF_EXISTS:${dep}>,${dep}>:$<TARGET_OBJECTS:${dep}>>)
    string(FIND ${dep} "." last_dot_loc REVERSE)
    math(EXPR name_loc "${last_dot_loc} + 1")
    string(SUBSTRING ${dep} ${name_loc} -1 name)
    if(LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
      set(prefix --image=arch=generic,triple=nvptx64-nvidia-cuda,feature=+ptx63)
    elseif(LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
      set(prefix --image=arch=generic,triple=amdgcn-amd-amdhsa)
    endif()

    # Use the 'clang-offload-packager' to merge these files into a binary blob.
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/binary/${name}.gpubin"
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/binary
      COMMAND ${LIBC_CLANG_OFFLOAD_PACKAGER}
              "${prefix},file=$<JOIN:${object},,file=>" -o
              ${CMAKE_CURRENT_BINARY_DIR}/binary/${name}.gpubin
      DEPENDS ${dep} ${base_target_name}
      COMMENT "Packaging LLVM offloading binary for '${object}'"
    )
    add_custom_target(${dep}.__gpubin__ DEPENDS ${dep}
                      "${CMAKE_CURRENT_BINARY_DIR}/binary/${name}.gpubin")

    # CMake does not permit setting the name on object files. In order to have
    # human readable names we create an empty stub file with the entrypoint
    # name. This empty file will then have the created binary blob embedded.
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/stubs/${name}.cpp"
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/stubs
      COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/stubs/${name}.cpp
      DEPENDS ${dep} ${dep}.__gpubin__ ${base_target_name}
    )
    add_custom_target(${dep}.__stub__
                      DEPENDS ${dep}.__gpubin__ "${CMAKE_CURRENT_BINARY_DIR}/stubs/${name}.cpp")

    add_library(${dep}.__fatbin__
      EXCLUDE_FROM_ALL OBJECT
      "${CMAKE_CURRENT_BINARY_DIR}/stubs/${name}.cpp"
    )

    # This is always compiled for the LLVM host triple instead of the native GPU
    # triple that is used by default in the build.
    target_compile_options(${dep}.__fatbin__ BEFORE PRIVATE -nostdlib)
    target_compile_options(${dep}.__fatbin__ PRIVATE
      --target=${LLVM_HOST_TRIPLE}
      "SHELL:-Xclang -fembed-offload-object=${CMAKE_CURRENT_BINARY_DIR}/binary/${name}.gpubin")
    add_dependencies(${dep}.__fatbin__
                     ${dep} ${dep}.__stub__ ${dep}.__gpubin__ ${base_target_name})

    # Set the list of newly create fat binaries containing embedded device code.
    list(APPEND objects $<TARGET_OBJECTS:${dep}.__fatbin__>)
  endforeach()

  add_library(
    ${target_name}
    STATIC
      ${objects}
  )
  set_target_properties(${target_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIBC_LIBRARY_DIR})
endfunction(add_gpu_entrypoint_library)

# A rule to build a library from a collection of entrypoint objects and bundle
# it in a single LLVM-IR bitcode file.
# Usage:
#     add_gpu_entrypoint_library(
#       DEPENDS <list of add_entrypoint_object targets>
#     )
function(add_bitcode_entrypoint_library target_name base_target_name)
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
  get_all_object_file_deps(all_deps "${fq_deps_list}")

  set(objects "")
  foreach(dep IN LISTS all_deps)
    set(object $<$<STREQUAL:$<TARGET_NAME_IF_EXISTS:${dep}>,${dep}>:$<TARGET_OBJECTS:${dep}>>)
    list(APPEND objects ${object})
  endforeach()

  set(output ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.bc)
  add_custom_command(
    OUTPUT ${output}
    COMMAND ${LIBC_LLVM_LINK} ${objects} -o ${output}
    DEPENDS ${all_deps} ${base_target_name}
    COMMENT "Linking LLVM-IR bitcode for ${base_target_name}"
    COMMAND_EXPAND_LISTS
  )
  add_custom_target(${target_name} DEPENDS ${output} ${all_deps})
  set_target_properties(${target_name} PROPERTIES TARGET_OBJECT ${output})
endfunction(add_bitcode_entrypoint_library)

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
  get_all_object_file_deps(all_deps "${fq_deps_list}")

  set(objects "")
  foreach(dep IN LISTS all_deps)
    list(APPEND objects $<$<STREQUAL:$<TARGET_NAME_IF_EXISTS:${dep}>,${dep}>:$<TARGET_OBJECTS:${dep}>>)
  endforeach(dep)

  add_library(
    ${target_name}
    STATIC
    ${objects}
  )
  set_target_properties(${target_name} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${LIBC_LIBRARY_DIR})
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
  set_target_properties(${target_name}  PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIBC_LIBRARY_DIR})
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
      if (NOT dep MATCHES "__copied_hdr__")
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
endfunction(create_header_library)

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
endfunction(add_header_library)
