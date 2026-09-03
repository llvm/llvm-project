add_header_library(
  abi_prefix
  HDRS
    abi_prefix.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.macros.properties.cpu_features
)

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

add_vector_math_entrypoint_object(acosf)
add_vector_math_entrypoint_object(acoshf)
add_vector_math_entrypoint_object(acospif)
add_vector_math_entrypoint_object(asinf)
add_vector_math_entrypoint_object(asinhf)
add_vector_math_entrypoint_object(asinpif)
add_vector_math_entrypoint_object(atanf)
add_vector_math_entrypoint_object(atanhf)
add_vector_math_entrypoint_object(cbrtf)
add_vector_math_entrypoint_object(cosf)
add_vector_math_entrypoint_object(coshf)
add_vector_math_entrypoint_object(cospif)
add_vector_math_entrypoint_object(erff)
add_vector_math_entrypoint_object(exp2f)
add_vector_math_entrypoint_object(exp2m1f)
add_vector_math_entrypoint_object(exp10f)
add_vector_math_entrypoint_object(exp10m1f)
add_vector_math_entrypoint_object(expf)
add_vector_math_entrypoint_object(expm1f)
add_vector_math_entrypoint_object(log10f)
add_vector_math_entrypoint_object(log1pf)
add_vector_math_entrypoint_object(log2f)
add_vector_math_entrypoint_object(logf)
add_vector_math_entrypoint_object(rsqrtf)
add_vector_math_entrypoint_object(sinf)
add_vector_math_entrypoint_object(sinhf)
add_vector_math_entrypoint_object(sinpif)
add_vector_math_entrypoint_object(sqrtf)
add_vector_math_entrypoint_object(tanf)
add_vector_math_entrypoint_object(tanhf)
add_vector_math_entrypoint_object(tanpif)
