function(add_startup_object name)
  cmake_parse_arguments(
    "ADD_STARTUP_OBJECT"
    "ALIAS" # Option argument
    "SRC"   # Single value arguments
    "DEPENDS;COMPILE_OPTIONS" # Multi value arguments
    ${ARGN}
  )

  get_fq_target_name(${name} fq_target_name)
  if(ADD_STARTUP_OBJECT_ALIAS)
    get_fq_deps_list(fq_dep_list ${ADD_STARTUP_OBJECT_DEPENDS})
    add_library(${fq_target_name} ALIAS ${fq_dep_list})
    return()
  endif()

  add_object_library(
    ${name}
    SRCS ${ADD_STARTUP_OBJECT_SRC}
    COMPILE_OPTIONS ${ADD_STARTUP_OBJECT_COMPILE_OPTIONS}
    ${ADD_STARTUP_OBJECT_UNPARSED_ARGUMENTS}
    DEPENDS ${ADD_STARTUP_OBJECT_DEPENDS}
  )
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      OUTPUT_NAME ${name}.o
  )

  # Make an executable target of relocatable bitcode for clang if needed.
  if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR)
    add_executable(${fq_target_name}.exe $<TARGET_OBJECTS:${fq_target_name}>)
    set_target_properties(${fq_target_name}.exe PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${LIBC_LIBRARY_DIR}
      RUNTIME_OUTPUT_NAME ${name}.o)
    target_link_options(${fq_target_name}.exe PRIVATE
                        "-r" "-nostdlib" "-flto" "-Wl,--lto-emit-llvm")
  endif()
endfunction()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  add_subdirectory(${LIBC_TARGET_ARCHITECTURE})
endif()

add_startup_object(
  crt1
  ALIAS
  DEPENDS
  .${LIBC_TARGET_ARCHITECTURE}.crt1
)

add_custom_target(libc-startup)
set(startup_components crt1)
foreach(target IN LISTS startup_components)
  set(fq_target_name libc.startup.gpu.${target})
  add_dependencies(libc-startup ${fq_target_name})
  install(FILES $<TARGET_OBJECTS:${fq_target_name}>
          DESTINATION ${LIBC_INSTALL_LIBRARY_DIR}
          RENAME $<TARGET_PROPERTY:${fq_target_name},OUTPUT_NAME>
          COMPONENT libc)
endforeach()
