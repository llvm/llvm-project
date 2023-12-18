# Math functions not yet available in the libc project, or those not yet tuned
# for GPU workloads are provided as wrappers over vendor libraries. If we find
# them ahead of time we will import them statically. Otherwise, we will keep
# them as external references and expect them to be resolved by the user when
# they compile. In the future,we will use implementations from the 'libc'
# project and not provide these wrappers.
add_subdirectory(vendor)

# For the GPU we want to be able to optionally depend on the vendor libraries
# until we have a suitable replacement inside `libc`.
# TODO: We should have an option to enable or disable these on a per-function
# basis.
option(LIBC_GPU_VENDOR_MATH "Use vendor wrappers for GPU math" ON)
function(add_math_entrypoint_gpu_object name)
  get_fq_target_name("vendor.${name}" fq_vendor_specific_target_name)
  if(TARGET ${fq_vendor_specific_target_name} AND ${LIBC_GPU_VENDOR_MATH})
    return()
  endif()

  add_entrypoint_object(
    ${name}
    ${ARGN}
  )
endfunction()

add_math_entrypoint_gpu_object(
  ceil
  SRCS
    ceil.cpp
  HDRS
    ../ceil.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  ceilf
  SRCS
    ceilf.cpp
  HDRS
    ../ceilf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  copysign
  SRCS
    copysign.cpp
  HDRS
    ../copysign.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  copysignf
  SRCS
    copysignf.cpp
  HDRS
    ../copysignf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  fabs
  SRCS
    fabs.cpp
  HDRS
    ../fabs.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  fabsf
  SRCS
    fabsf.cpp
  HDRS
    ../fabsf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  floor
  SRCS
    floor.cpp
  HDRS
    ../floor.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  floorf
  SRCS
    floorf.cpp
  HDRS
    ../floorf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  fma
  SRCS
    fma.cpp
  HDRS
    ../fma.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  fmaf
  SRCS
    fmaf.cpp
  HDRS
    ../fmaf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  fmax
  SRCS
    fmax.cpp
  HDRS
    ../fmax.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  fmaxf
  SRCS
    fmaxf.cpp
  HDRS
    ../fmaxf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  fmin
  SRCS
    fmin.cpp
  HDRS
    ../fmin.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  fminf
  SRCS
    fminf.cpp
  HDRS
    ../fminf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  fmod
  SRCS
    fmod.cpp
  HDRS
    ../fmod.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  fmodf
  SRCS
    fmodf.cpp
  HDRS
    ../fmodf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  lround
  SRCS
    lround.cpp
  HDRS
    ../lround.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  lroundf
  SRCS
    lroundf.cpp
  HDRS
    ../lroundf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  llround
  SRCS
    llround.cpp
  HDRS
    ../llround.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  llroundf
  SRCS
    llroundf.cpp
  HDRS
    ../llroundf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  modf
  SRCS
    modf.cpp
  HDRS
    ../modf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  modff
  SRCS
    modff.cpp
  HDRS
    ../modff.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  nearbyint
  SRCS
    nearbyint.cpp
  HDRS
    ../nearbyint.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  nearbyintf
  SRCS
    nearbyintf.cpp
  HDRS
    ../nearbyintf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  remainder
  SRCS
    remainder.cpp
  HDRS
    ../remainder.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  remainderf
  SRCS
    remainderf.cpp
  HDRS
    ../remainderf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  rint
  SRCS
    rint.cpp
  HDRS
    ../rint.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  rintf
  SRCS
    rintf.cpp
  HDRS
    ../rintf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  round
  SRCS
    round.cpp
  HDRS
    ../round.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  sinh
  SRCS
    sinh.cpp
  HDRS
    ../sinh.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  sqrt
  SRCS
    sqrt.cpp
  HDRS
    ../sqrt.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  sqrtf
  SRCS
    sqrtf.cpp
  HDRS
    ../sqrtf.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  tan
  SRCS
    tan.cpp
  HDRS
    ../tan.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  tanh
  SRCS
    tanh.cpp
  HDRS
    ../tanh.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  trunc
  SRCS
    trunc.cpp
  HDRS
    ../trunc.h
  COMPILE_OPTIONS
    -O2
)

add_math_entrypoint_gpu_object(
  truncf
  SRCS
    truncf.cpp
  HDRS
    ../truncf.h
  COMPILE_OPTIONS
    -O2
)
