function(add_loader_object name)
  cmake_parse_arguments(
    "ADD_LOADER_OBJECT"
    "ALIAS" # Option argument
    "SRC"   # Single value arguments
    "DEPENDS;COMPILE_OPTIONS" # Multi value arguments
    ${ARGN}
  )

  get_fq_target_name(${name} fq_target_name)
  get_fq_deps_list(fq_deps_list ${ADD_LOADER_OBJECT_DEPENDS})
  if(ADD_LOADER_OBJECT_ALIAS)
    list(LENGTH ADD_LOADER_OBJECT_DEPENDS deps_size)
    if(NOT (${deps_size} EQUAL "1"))
      message(FATAL_ERROR "A loader object alias should have exactly one dependency.")
    endif()
    list(GET ADD_LOADER_OBJECT_DEPENDS 0 dep)
    get_fq_dep_name(fq_dep_name ${dep})

    add_custom_target(${fq_target_name})
    add_dependencies(${fq_target_name} ${fq_dep_name})
    get_target_property(dep_objfile ${fq_dep_name} OBJECT_FILES)
    set_target_properties(
      ${fq_target_name}
      PROPERTIES
        "TARGET_TYPE" "${OBJECT_LIBRARY_TARGET_TYPE}"
        "OBJECT_FILES" ""
        "DEPS" "${fq_dep_name}"
    )
    return()
  endif()

  add_object_library(
    ${name}.__objects__
    SRCS ${ADD_LOADER_OBJECT_SRC}
    DEPENDS ${ADD_LOADER_OBJECT_DEPENDS}
    COMPILE_OPTIONS ${ADD_LOADER_OBJECT_COMPILE_OPTIONS}
  )

  set(objfile ${LIBC_BUILD_DIR}/lib/${name}.o)
  add_custom_command(
    OUTPUT ${objfile}
    COMMAND cp $<TARGET_OBJECTS:${fq_target_name}.__objects__> ${objfile}
    DEPENDS $<TARGET_OBJECTS:${fq_target_name}.__objects__>
  )
  add_custom_target(
    ${fq_target_name}
    DEPENDS ${objfile}
  )
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      "TARGET_TYPE" "${OBJECT_LIBRARY_TARGET_TYPE}"
      "OBJECT_FILES" ""
      "DEPS" "${fq_target_name}.__objects__"
  )
endfunction()

if(NOT (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_MACHINE}))
  message(STATUS "Skipping loader for target machine ${LIBC_TARGET_MACHINE}")
  return()
endif()

add_subdirectory(${LIBC_TARGET_MACHINE})

add_loader_object(
  crt1
  ALIAS
  DEPENDS
    .${LIBC_TARGET_MACHINE}.crt1
)
