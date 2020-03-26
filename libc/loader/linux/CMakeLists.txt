function(add_loader_object name)
  cmake_parse_arguments(
    "ADD_LOADER_OBJECT"
    ""    # No option arguments
    "SRC" # Single value arguments
    "DEPENDS;COMPILE_OPTIONS" # Multi value arguments
    ${ARGN}
  )
  add_object(
    ${name}_object
    SRC ${ADD_LOADER_OBJECT_SRC}
    DEPENDS ${ADD_LOADER_OBJECT_DEPENDS}
    COMPILE_OPTIONS ${ADD_LOADER_OBJECT_COMPILE_OPTIONS}
  )

  set(objfile ${LIBC_BUILD_DIR}/lib/${name}.o)
  add_custom_command(
    OUTPUT ${objfile}
    COMMAND cp $<TARGET_OBJECTS:${name}_object> ${objfile}
    DEPENDS $<TARGET_OBJECTS:${name}_object>
  )
  add_custom_target(
    ${name}
    DEPENDS ${objfile}
  )
  set_target_properties(
    ${name}
    PROPERTIES
      "TARGET_TYPE" "LOADER_OBJECT"
      "OBJECT_FILE" ${objfile}
  )
endfunction()

add_subdirectory(${LIBC_TARGET_MACHINE})
