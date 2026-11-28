add_subdirectory(generic)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  add_subdirectory(${LIBC_TARGET_ARCHITECTURE})
endif()

function(add_complex_entrypoint_object name)
  # We prefer machine specific implementation if available. Hence we check
  # that first and return early if we are able to add an alias target for the
  # machine specific implementation.
  get_fq_target_name("${LIBC_TARGET_ARCHITECTURE}.${name}" fq_machine_specific_target_name)
  if(TARGET ${fq_machine_specific_target_name})
    add_entrypoint_object(
      ${name}
      ALIAS
      DEPENDS
        .${LIBC_TARGET_ARCHITECTURE}.${name}
    )
    return()
  endif()

  get_fq_target_name("generic.${name}" fq_generic_target_name)
  if(TARGET ${fq_generic_target_name})
    add_entrypoint_object(
      ${name}
      ALIAS
      DEPENDS
        .generic.${name}
    )
    return()
  endif()
endfunction()

add_complex_entrypoint_object(creal)
add_complex_entrypoint_object(crealf)
add_complex_entrypoint_object(creall)
add_complex_entrypoint_object(crealf16)
add_complex_entrypoint_object(crealf128)

add_complex_entrypoint_object(cimag)
add_complex_entrypoint_object(cimagf)
add_complex_entrypoint_object(cimagl)
add_complex_entrypoint_object(cimagf16)
add_complex_entrypoint_object(cimagf128)

add_complex_entrypoint_object(conj)
add_complex_entrypoint_object(conjf)
add_complex_entrypoint_object(conjl)
add_complex_entrypoint_object(conjf16)
add_complex_entrypoint_object(conjf128)

add_complex_entrypoint_object(cproj)
add_complex_entrypoint_object(cprojf)
add_complex_entrypoint_object(cprojl)
add_complex_entrypoint_object(cprojf16)
add_complex_entrypoint_object(cprojf128)

add_complex_entrypoint_object(cabs)
add_complex_entrypoint_object(cabsf)
add_complex_entrypoint_object(carg)
add_complex_entrypoint_object(cargf)
