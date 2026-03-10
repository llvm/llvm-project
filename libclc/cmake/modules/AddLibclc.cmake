# Converts a list of relative source paths to absolute paths and exports
# it to the parent scope.
macro(libclc_configure_source_list variable path)
  set(${variable} ${ARGN})
  list(TRANSFORM ${variable} PREPEND "${path}/")
  set(${variable} ${${variable}} PARENT_SCOPE)
endmacro()

# Appends a compile option to the given source files. Paths are relative
# to `path` and the property is set in the top-level libclc directory scope.
macro(libclc_configure_source_options path option)
  set(_option_srcs ${ARGN})
  list(TRANSFORM _option_srcs PREPEND "${path}/")
  set_property(SOURCE ${_option_srcs}
    DIRECTORY ${LIBCLC_SOURCE_DIR}
    APPEND PROPERTY COMPILE_OPTIONS ${option}
  )
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

# Links one or more libclc builtin libraries together, optionally
# internalizing dependencies, then produces a final .bc or .spv file.
function(link_libclc_builtin_library target_name)
  cmake_parse_arguments(ARG
    ""
    "ARCH;TRIPLE;FOLDER"
    "LIBRARIES;INTERNALIZE_LIBRARIES;OPT_FLAGS"
    ${ARGN}
  )

  set(library_dir ${LIBCLC_OUTPUT_LIBRARY_DIR}/${ARG_TRIPLE})
  file(MAKE_DIRECTORY ${library_dir})

  set(linked_bc ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.linked.bc)

  set(link_cmd ${llvm-link_exe})
  foreach(lib ${ARG_LIBRARIES})
    list(APPEND link_cmd $<TARGET_FILE:${lib}>)
  endforeach()
  if(ARG_INTERNALIZE_LIBRARIES)
    list(APPEND link_cmd --internalize --only-needed)
    foreach(lib ${ARG_INTERNALIZE_LIBRARIES})
      list(APPEND link_cmd $<TARGET_FILE:${lib}>)
    endforeach()
  endif()
  list(APPEND link_cmd -o ${linked_bc})

  set(link_deps ${llvm-link_target}
    ${ARG_LIBRARIES} ${ARG_INTERNALIZE_LIBRARIES}
  )
  add_custom_command(OUTPUT ${linked_bc}
    COMMAND ${link_cmd}
    DEPENDS ${link_deps}
  )

  if(ARG_ARCH STREQUAL spirv OR ARG_ARCH STREQUAL spirv64)
    # SPIR-V targets produce a .spv file from the linked bitcode.
    set(builtins_lib ${library_dir}/libclc.spv)
    if(LIBCLC_USE_SPIRV_BACKEND)
      add_custom_command(OUTPUT ${builtins_lib}
        COMMAND ${CMAKE_CLC_COMPILER} -c --target=${ARG_TRIPLE}
                -mllvm --spirv-ext=+SPV_KHR_fma
                -x ir -o ${builtins_lib} ${linked_bc}
        DEPENDS ${linked_bc}
      )
    else()
      add_custom_command(OUTPUT ${builtins_lib}
        COMMAND ${llvm-spirv_exe}
                --spirv-max-version=1.1
                --spirv-ext=+SPV_KHR_fma
                -o ${builtins_lib} ${linked_bc}
        DEPENDS ${llvm-spirv_target} ${linked_bc}
      )
    endif()
  else()
    # All other targets produce an optimized .bc file.
    set(builtins_lib ${library_dir}/libclc.bc)
    add_custom_command(OUTPUT ${builtins_lib}
      COMMAND ${opt_exe} ${ARG_OPT_FLAGS} -o ${builtins_lib} ${linked_bc}
      DEPENDS ${opt_target} ${linked_bc}
    )
  endif()

  add_custom_target(${target_name} ALL DEPENDS ${builtins_lib})
  set_target_properties(${target_name} PROPERTIES
    TARGET_FILE ${builtins_lib}
    FOLDER ${ARG_FOLDER}
  )
endfunction()

# Builds an OpenCL builtins library from sources, links it with any
# internalized dependencies via link_libclc_builtin_library, and adds
# a verification test for unresolved symbols.
function(add_libclc_library target_name)
  cmake_parse_arguments(ARG
    ""
    "ARCH;TRIPLE;TARGET_TRIPLE"
    "SOURCES;COMPILE_OPTIONS;INCLUDE_DIRS;COMPILE_DEFINITIONS;INTERNALIZE_LIBRARIES;OPT_FLAGS"
    ${ARGN}
  )

  set(opencl_lib ${target_name}_opencl_builtins)
  add_libclc_builtin_library(${opencl_lib}
    SOURCES ${ARG_SOURCES}
    COMPILE_OPTIONS ${ARG_COMPILE_OPTIONS}
    INCLUDE_DIRS ${ARG_INCLUDE_DIRS}
    COMPILE_DEFINITIONS ${ARG_COMPILE_DEFINITIONS}
    FOLDER "libclc/Device IR/OpenCL"
  )

  link_libclc_builtin_library(${target_name}
    ARCH ${ARG_ARCH}
    TRIPLE ${ARG_TRIPLE}
    LIBRARIES ${opencl_lib}
    INTERNALIZE_LIBRARIES ${ARG_INTERNALIZE_LIBRARIES}
    OPT_FLAGS ${ARG_OPT_FLAGS}
    FOLDER "libclc/Device IR/Library"
  )

  add_dependencies(libclc-opencl-builtins ${target_name})
  set(builtins_file $<TARGET_PROPERTY:${target_name},TARGET_FILE>)

  install(FILES ${builtins_file}
    DESTINATION ${LIBCLC_INSTALL_DIR}/${ARG_TRIPLE}
    COMPONENT libclc-opencl-builtins
  )

  # Verify there are no unresolved external functions in the library.
  if(NOT ARG_ARCH MATCHES "^(nvptx|clspv)(64)?$" AND
     NOT ARG_ARCH MATCHES "^spirv(64)?$")
    set(builtins_file $<TARGET_PROPERTY:${target_name},TARGET_FILE>)
    add_test(NAME external-funcs-${target_name}
      COMMAND ./check_external_funcs.sh
              ${builtins_file} ${LLVM_TOOLS_BINARY_DIR}
      WORKING_DIRECTORY ${LIBCLC_SOURCE_DIR})
  endif()
endfunction()
