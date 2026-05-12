# Adds source files to a libclc builtin library target with deduplication. If a
# source with the same basename already exists in the target's SOURCES property
# the new file is skipped. This enables target-specific directories to override
# generic implementations when they are included first.
#
# Sources are specified as paths relative to CMAKE_CURRENT_SOURCE_DIR, or
# relative to BASE_DIR if provided.
function(libclc_add_sources target)
  cmake_parse_arguments(ARG "" "BASE_DIR" "FILES" ${ARGN})
  if(NOT ARG_BASE_DIR)
    set(ARG_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  get_target_property(existing ${target} SOURCES)

  set(seen)
  foreach(file IN LISTS existing)
    get_filename_component(name "${file}" NAME)
    list(APPEND seen "${name}")
  endforeach()

  set(new_sources)
  foreach(rel_src IN LISTS ARG_FILES)
    get_filename_component(name "${rel_src}" NAME)
    if(NOT name IN_LIST seen)
      list(APPEND new_sources "${ARG_BASE_DIR}/${rel_src}")
      list(APPEND seen "${name}")
    endif()
  endforeach()

  if(new_sources)
    target_sources(${target} PRIVATE ${new_sources})
    set(inc_dirs)
    foreach(file IN LISTS new_sources)
      get_filename_component(dir "${file}" DIRECTORY)
      list(APPEND inc_dirs "${dir}")
    endforeach()
    list(REMOVE_DUPLICATES inc_dirs)
    target_include_directories(${target} PRIVATE ${inc_dirs})
  endif()
endfunction()

# Appends a compile option to the given source files. Source paths are
# relative to CMAKE_CURRENT_SOURCE_DIR. The property is set in the
# top-level libclc directory scope.
function(libclc_set_source_options option)
  set(srcs ${ARGN})
  list(TRANSFORM srcs PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
  set_property(SOURCE ${srcs}
    DIRECTORY ${LIBCLC_SOURCE_DIR}
    APPEND PROPERTY COMPILE_OPTIONS ${option}
  )
endfunction()

# Creates a static library target for libclc builtins and configures its
# compile options, include directories, and definitions. Subdirectories
# populate sources via libclc_add_sources() after this call.
function(libclc_add_builtin_library target_name)
  cmake_parse_arguments(ARG
    ""
    "FOLDER"
    "COMPILE_OPTIONS;INCLUDE_DIRS;COMPILE_DEFINITIONS"
    ${ARGN}
  )

  add_library(${target_name} STATIC)
  target_compile_options(${target_name} PRIVATE ${ARG_COMPILE_OPTIONS})
  target_include_directories(${target_name} PRIVATE ${ARG_INCLUDE_DIRS})
  target_compile_definitions(${target_name} PRIVATE ${ARG_COMPILE_DEFINITIONS})
  set_target_properties(${target_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    FOLDER ${ARG_FOLDER}
  )
endfunction()

# Links one or more libclc builtin libraries together, optionally
# internalizing dependencies, then produces a final .bc or .spv file.
function(libclc_link_library target_name)
  cmake_parse_arguments(ARG
    ""
    "ARCH;TRIPLE;TARGET_TRIPLE;FOLDER;OUTPUT_FILENAME"
    "LIBRARIES;INTERNALIZE_LIBRARIES;OPT_FLAGS"
    ${ARGN}
  )

  if(NOT ARG_OUTPUT_FILENAME)
    message(FATAL_ERROR "OUTPUT_FILENAME is required for libclc_link_library")
  endif()
  if(NOT ARG_LIBRARIES)
    message(FATAL_ERROR "LIBRARIES is required for libclc_link_library")
  endif()

  set(library_dir ${LIBCLC_OUTPUT_LIBRARY_DIR}/${ARG_TARGET_TRIPLE})
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

  string(REPLACE "-" ";" triple_parts "${ARG_TRIPLE}")
  list(GET triple_parts 2 triple_os)
  if(ARG_ARCH IN_LIST LIBCLC_ARCHS_SPIRV AND NOT triple_os STREQUAL vulkan)
    # SPIR-V targets produce a .spv file from the linked bitcode.
    set(builtins_lib ${library_dir}/${ARG_OUTPUT_FILENAME}.spv)
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
        DEPENDS ${linked_bc}
      )
    endif()
  else()
    # All other targets produce an optimized .bc file.
    set(builtins_lib ${library_dir}/${ARG_OUTPUT_FILENAME}.bc)
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

# Links builtin library targets, produces the final output file, and
# registers it for installation.
function(libclc_add_library target_name)
  cmake_parse_arguments(ARG
    ""
    "ARCH;TRIPLE;TARGET_TRIPLE;OUTPUT_FILENAME;PARENT_TARGET"
    "LIBRARIES;INTERNALIZE_LIBRARIES;OPT_FLAGS"
    ${ARGN}
  )

  if(NOT ARG_OUTPUT_FILENAME)
    message(FATAL_ERROR "OUTPUT_FILENAME is required for libclc_add_library")
  endif()
  if(NOT ARG_PARENT_TARGET)
    message(FATAL_ERROR "PARENT_TARGET is required for libclc_add_library")
  endif()
  if(NOT ARG_LIBRARIES)
    message(FATAL_ERROR "LIBRARIES is required for libclc_add_library")
  endif()

  libclc_link_library(${target_name}
    ARCH ${ARG_ARCH}
    TRIPLE ${ARG_TRIPLE}
    TARGET_TRIPLE ${ARG_TARGET_TRIPLE}
    LIBRARIES ${ARG_LIBRARIES}
    INTERNALIZE_LIBRARIES ${ARG_INTERNALIZE_LIBRARIES}
    OPT_FLAGS ${ARG_OPT_FLAGS}
    OUTPUT_FILENAME "${ARG_OUTPUT_FILENAME}"
    FOLDER "libclc/Device IR/Library"
  )

  add_dependencies(${ARG_PARENT_TARGET} ${target_name})
  set(builtins_file $<TARGET_PROPERTY:${target_name},TARGET_FILE>)

  install(FILES ${builtins_file}
    DESTINATION ${LIBCLC_INSTALL_DIR}/${ARG_TARGET_TRIPLE}
    COMPONENT ${ARG_PARENT_TARGET}
  )
endfunction()
