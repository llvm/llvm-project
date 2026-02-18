add_subdirectory(generic)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  add_subdirectory(${LIBC_TARGET_ARCHITECTURE})
endif()

function(add_vector_math_entrypoint_object name)
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

  # Add a dummy entrypoint object for missing implementations. They will be skipped
  # anyway as there will be no entry for them in the target entrypoints list.
  add_entrypoint_object(
    ${name}
    SRCS
      dummy_srcs
    HDRS
      dummy_hdrs
  )
endfunction()

add_vector_math_entrypoint_object(expf)
