add_subdirectory(generic)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_MACHINE})
  add_subdirectory(${LIBC_TARGET_MACHINE})
endif()

function(add_math_entrypoint_object name)
  # We prefer machine specific implementation if available. Hence we check
  # that first and retrun early if we are able to add an alias target for the
  # machine specific implementation.
  get_fq_target_name("${LIBC_TARGET_MACHINE}.${name}" fq_machine_specific_target_name)
  if(TARGET ${fq_machine_specific_target_name})
    add_entrypoint_object(
      ${name}
      ALIAS
      DEPENDS
        .${LIBC_TARGET_MACHINE}.${name}
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

  message(FATAL_ERROR "No machine specific or generic implementation found for ${name}.")
endfunction()

add_math_entrypoint_object(ceil)
add_math_entrypoint_object(ceilf)
add_math_entrypoint_object(ceill)

add_math_entrypoint_object(copysign)
add_math_entrypoint_object(copysignf)
add_math_entrypoint_object(copysignl)

add_math_entrypoint_object(cosf)

add_math_entrypoint_object(expf)

add_math_entrypoint_object(exp2f)

add_math_entrypoint_object(fabs)
add_math_entrypoint_object(fabsf)
add_math_entrypoint_object(fabsl)

add_math_entrypoint_object(fdim)
add_math_entrypoint_object(fdimf)
add_math_entrypoint_object(fdiml)

add_math_entrypoint_object(floor)
add_math_entrypoint_object(floorf)
add_math_entrypoint_object(floorl)

add_math_entrypoint_object(fmaf)

add_math_entrypoint_object(fmax)
add_math_entrypoint_object(fmaxf)
add_math_entrypoint_object(fmaxl)

add_math_entrypoint_object(fmin)
add_math_entrypoint_object(fminf)
add_math_entrypoint_object(fminl)

add_math_entrypoint_object(frexp)
add_math_entrypoint_object(frexpf)
add_math_entrypoint_object(frexpl)

add_math_entrypoint_object(hypot)
add_math_entrypoint_object(hypotf)

add_math_entrypoint_object(ilogb)
add_math_entrypoint_object(ilogbf)
add_math_entrypoint_object(ilogbl)

add_math_entrypoint_object(ldexp)
add_math_entrypoint_object(ldexpf)
add_math_entrypoint_object(ldexpl)

add_math_entrypoint_object(logb)
add_math_entrypoint_object(logbf)
add_math_entrypoint_object(logbl)

add_math_entrypoint_object(llrint)
add_math_entrypoint_object(llrintf)
add_math_entrypoint_object(llrintl)

add_math_entrypoint_object(llround)
add_math_entrypoint_object(llroundf)
add_math_entrypoint_object(llroundl)

add_math_entrypoint_object(lrint)
add_math_entrypoint_object(lrintf)
add_math_entrypoint_object(lrintl)

add_math_entrypoint_object(lround)
add_math_entrypoint_object(lroundf)
add_math_entrypoint_object(lroundl)

add_math_entrypoint_object(modf)
add_math_entrypoint_object(modff)
add_math_entrypoint_object(modfl)

add_math_entrypoint_object(nearbyint)
add_math_entrypoint_object(nearbyintf)
add_math_entrypoint_object(nearbyintl)

add_math_entrypoint_object(nextafter)
add_math_entrypoint_object(nextafterf)
add_math_entrypoint_object(nextafterl)

add_math_entrypoint_object(remainder)
add_math_entrypoint_object(remainderf)
add_math_entrypoint_object(remainderl)

add_math_entrypoint_object(remquo)
add_math_entrypoint_object(remquof)
add_math_entrypoint_object(remquol)

add_math_entrypoint_object(rint)
add_math_entrypoint_object(rintf)
add_math_entrypoint_object(rintl)

add_math_entrypoint_object(round)
add_math_entrypoint_object(roundf)
add_math_entrypoint_object(roundl)

add_math_entrypoint_object(sincosf)

add_math_entrypoint_object(sinf)

add_math_entrypoint_object(sqrt)
add_math_entrypoint_object(sqrtf)
add_math_entrypoint_object(sqrtl)

add_math_entrypoint_object(trunc)
add_math_entrypoint_object(truncf)
add_math_entrypoint_object(truncl)
