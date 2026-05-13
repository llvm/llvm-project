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

# Creates an object library target for libclc builtins and configures its
# compile options, include directories, and definitions. Subdirectories
# populate sources via libclc_add_sources() after this call.
function(libclc_add_builtin_library target_name)
  cmake_parse_arguments(ARG
    ""
    "FOLDER"
    "COMPILE_OPTIONS;INCLUDE_DIRS;COMPILE_DEFINITIONS"
    ${ARGN}
  )

  add_library(${target_name} OBJECT)
  target_compile_options(${target_name} PRIVATE ${ARG_COMPILE_OPTIONS})
  target_include_directories(${target_name} PRIVATE ${ARG_INCLUDE_DIRS})
  target_compile_definitions(${target_name} PRIVATE ${ARG_COMPILE_DEFINITIONS})
  set_target_properties(${target_name} PROPERTIES FOLDER ${ARG_FOLDER})
endfunction()

# Links builtin libclc object libraries together into a merged bitcode or SPIR-V
# file and a static archive.
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

  set(library_dir ${LIBCLC_OUTPUT_LIBRARY_DIR}/${ARG_TARGET_TRIPLE})
  file(MAKE_DIRECTORY ${library_dir})

  # Create a combined static archive from all object libraries for installation.
  set(archive_target ${target_name}.a)
  add_library(${archive_target} STATIC)
  target_link_libraries(${archive_target} PRIVATE
    ${ARG_LIBRARIES} ${ARG_INTERNALIZE_LIBRARIES}
  )
  set_target_properties(${archive_target} PROPERTIES
    OUTPUT_NAME ${ARG_OUTPUT_FILENAME}
    PREFIX ""
    ARCHIVE_OUTPUT_DIRECTORY ${library_dir}
    FOLDER "libclc/Device IR/Library"
  )

  # Link the object files, using a temporary library as for internalization.
  set(linked_bc ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.linked.bc)
  set(link_cmd ${llvm-link_exe})
  set(link_deps ${llvm-link_target})
  foreach(lib ${ARG_LIBRARIES})
    add_library(${target_name}-${lib} STATIC)
    target_link_libraries(${target_name}-${lib} PRIVATE ${lib})
    set_target_properties(${target_name}-${lib} PROPERTIES
      FOLDER "libclc/Device IR/Library"
    )
    list(APPEND link_cmd $<TARGET_FILE:${target_name}-${lib}>)
    list(APPEND link_deps ${target_name}-${lib})
  endforeach()
  if(ARG_INTERNALIZE_LIBRARIES)
    list(APPEND link_cmd --internalize --only-needed)
    foreach(lib ${ARG_INTERNALIZE_LIBRARIES})
      add_library(${target_name}-${lib} STATIC)
      target_link_libraries(${target_name}-${lib} PRIVATE ${lib})
      set_target_properties(${target_name}-${lib} PROPERTIES
        FOLDER "libclc/Device IR/Library"
      )
      list(APPEND link_cmd $<TARGET_FILE:${target_name}-${lib}>)
      list(APPEND link_deps ${target_name}-${lib})
    endforeach()
  endif()
  list(APPEND link_cmd -o ${linked_bc})

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
    FOLDER "libclc/Device IR/Library"
  )

  add_dependencies(${ARG_PARENT_TARGET} ${target_name} ${archive_target})

  install(FILES ${builtins_lib} $<TARGET_FILE:${archive_target}>
    DESTINATION ${LIBCLC_INSTALL_DIR}/${ARG_TARGET_TRIPLE}
    COMPONENT ${ARG_PARENT_TARGET}
  )
endfunction()
