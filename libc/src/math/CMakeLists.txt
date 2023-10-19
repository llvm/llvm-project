add_subdirectory(generic)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  add_subdirectory(${LIBC_TARGET_ARCHITECTURE})
endif()

function(add_math_entrypoint_object name)
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

  # The GPU optionally depends on vendor libraries. If we emitted one of these
  # entrypoints it means the user requested it and we should use it instead.
  get_fq_target_name("${LIBC_TARGET_ARCHITECTURE}.vendor.${name}" fq_vendor_specific_target_name)
  if(TARGET ${fq_vendor_specific_target_name})
    add_entrypoint_object(
      ${name}
      ALIAS
      DEPENDS
        .${LIBC_TARGET_ARCHITECTURE}.vendor.${name}
      VENDOR
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

add_math_entrypoint_object(acos)
add_math_entrypoint_object(acosf)
add_math_entrypoint_object(acosh)
add_math_entrypoint_object(acoshf)

add_math_entrypoint_object(asin)
add_math_entrypoint_object(asinf)
add_math_entrypoint_object(asinh)
add_math_entrypoint_object(asinhf)

add_math_entrypoint_object(atan)
add_math_entrypoint_object(atanf)

add_math_entrypoint_object(atan2)
add_math_entrypoint_object(atan2f)

add_math_entrypoint_object(atanh)
add_math_entrypoint_object(atanhf)

add_math_entrypoint_object(ceil)
add_math_entrypoint_object(ceilf)
add_math_entrypoint_object(ceill)

add_math_entrypoint_object(copysign)
add_math_entrypoint_object(copysignf)
add_math_entrypoint_object(copysignl)

add_math_entrypoint_object(cos)
add_math_entrypoint_object(cosf)
add_math_entrypoint_object(cosh)
add_math_entrypoint_object(coshf)

add_math_entrypoint_object(erf)
add_math_entrypoint_object(erff)

add_math_entrypoint_object(exp)
add_math_entrypoint_object(expf)

add_math_entrypoint_object(exp2)
add_math_entrypoint_object(exp2f)

add_math_entrypoint_object(exp10)
add_math_entrypoint_object(exp10f)

add_math_entrypoint_object(expm1)
add_math_entrypoint_object(expm1f)

add_math_entrypoint_object(fabs)
add_math_entrypoint_object(fabsf)
add_math_entrypoint_object(fabsl)

add_math_entrypoint_object(fdim)
add_math_entrypoint_object(fdimf)
add_math_entrypoint_object(fdiml)

add_math_entrypoint_object(floor)
add_math_entrypoint_object(floorf)
add_math_entrypoint_object(floorl)

add_math_entrypoint_object(fma)
add_math_entrypoint_object(fmaf)

add_math_entrypoint_object(fmax)
add_math_entrypoint_object(fmaxf)
add_math_entrypoint_object(fmaxl)

add_math_entrypoint_object(fmin)
add_math_entrypoint_object(fminf)
add_math_entrypoint_object(fminl)

add_math_entrypoint_object(fmod)
add_math_entrypoint_object(fmodf)

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

add_math_entrypoint_object(log10)
add_math_entrypoint_object(log10f)

add_math_entrypoint_object(log1p)
add_math_entrypoint_object(log1pf)

add_math_entrypoint_object(log2)
add_math_entrypoint_object(log2f)

add_math_entrypoint_object(log)
add_math_entrypoint_object(logf)

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

add_math_entrypoint_object(pow)
add_math_entrypoint_object(powf)

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

add_math_entrypoint_object(scalbn)
add_math_entrypoint_object(scalbnf)
add_math_entrypoint_object(scalbnl)

add_math_entrypoint_object(sincos)
add_math_entrypoint_object(sincosf)

add_math_entrypoint_object(sin)
add_math_entrypoint_object(sinf)

add_math_entrypoint_object(sinh)
add_math_entrypoint_object(sinhf)

add_math_entrypoint_object(sqrt)
add_math_entrypoint_object(sqrtf)
add_math_entrypoint_object(sqrtl)

add_math_entrypoint_object(tan)
add_math_entrypoint_object(tanf)

add_math_entrypoint_object(tanh)
add_math_entrypoint_object(tanhf)

add_math_entrypoint_object(tgamma)
add_math_entrypoint_object(tgammaf)

add_math_entrypoint_object(trunc)
add_math_entrypoint_object(truncf)
add_math_entrypoint_object(truncl)
