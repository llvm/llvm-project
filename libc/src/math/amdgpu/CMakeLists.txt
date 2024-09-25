# Math functions not yet available in the libc project, or those not yet tuned
# for GPU workloads are provided as wrappers over vendor libraries. If we find
# them ahead of time we will import them statically. Otherwise, we will keep
# them as external references and expect them to be resolved by the user when
# they compile. In the future,we will use implementations from the 'libc'
# project and not provide these wrappers.
find_package(AMDDeviceLibs QUIET HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
if(AMDDeviceLibs_FOUND)
  message(STATUS "Found the ROCm device library. Implementations falling back "
                 "to the vendor libraries will be resolved statically.")
  get_target_property(ocml_path ocml IMPORTED_LOCATION)
  set(bitcode_link_flags
      "SHELL:-Xclang -mlink-builtin-bitcode -Xclang ${ocml_path}")
else()
  message(STATUS "Could not find the ROCm device library. Unimplemented "
                 "functions will be an external reference to the vendor libraries.")
endif()

add_entrypoint_object(
  ceil
  SRCS
    ceil.cpp
  HDRS
    ../ceil.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  ceilf
  SRCS
    ceilf.cpp
  HDRS
    ../ceilf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  copysign
  SRCS
    copysign.cpp
  HDRS
    ../copysign.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  copysignf
  SRCS
    copysignf.cpp
  HDRS
    ../copysignf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fabs
  SRCS
    fabs.cpp
  HDRS
    ../fabs.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fabsf
  SRCS
    fabsf.cpp
  HDRS
    ../fabsf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  floor
  SRCS
    floor.cpp
  HDRS
    ../floor.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  floorf
  SRCS
    floorf.cpp
  HDRS
    ../floorf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fma
  SRCS
    fma.cpp
  HDRS
    ../fma.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fmaf
  SRCS
    fmaf.cpp
  HDRS
    ../fmaf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fmax
  SRCS
    fmax.cpp
  HDRS
    ../fmax.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fmaxf
  SRCS
    fmaxf.cpp
  HDRS
    ../fmaxf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fmin
  SRCS
    fmin.cpp
  HDRS
    ../fmin.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fminf
  SRCS
    fminf.cpp
  HDRS
    ../fminf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fmod
  SRCS
    fmod.cpp
  HDRS
    ../fmod.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  fmodf
  SRCS
    fmodf.cpp
  HDRS
    ../fmodf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  nearbyint
  SRCS
    nearbyint.cpp
  HDRS
    ../nearbyint.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  nearbyintf
  SRCS
    nearbyintf.cpp
  HDRS
    ../nearbyintf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  remainder
  SRCS
    remainder.cpp
  HDRS
    ../remainder.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  remainderf
  SRCS
    remainderf.cpp
  HDRS
    ../remainderf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  rint
  SRCS
    rint.cpp
  HDRS
    ../rint.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  rintf
  SRCS
    rintf.cpp
  HDRS
    ../rintf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  round
  SRCS
    round.cpp
  HDRS
    ../round.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  sqrt
  SRCS
    sqrt.cpp
  HDRS
    ../sqrt.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  sqrtf
  SRCS
    sqrtf.cpp
  HDRS
    ../sqrtf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  trunc
  SRCS
    trunc.cpp
  HDRS
    ../trunc.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  truncf
  SRCS
    truncf.cpp
  HDRS
    ../truncf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  frexp
  SRCS
    frexp.cpp
  HDRS
    ../frexp.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  frexpf
  SRCS
    frexpf.cpp
  HDRS
    ../frexpf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  scalbn
  SRCS
    scalbn.cpp
  HDRS
    ../scalbn.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  scalbnf
  SRCS
    scalbnf.cpp
  HDRS
    ../scalbnf.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  ldexp
  SRCS
    ldexp.cpp
  HDRS
    ../ldexp.h
  COMPILE_OPTIONS
    -O2
)

add_entrypoint_object(
  ldexpf
  SRCS
    ldexpf.cpp
  HDRS
    ../ldexpf.h
  COMPILE_OPTIONS
    -O2
)

# The following functions currently are not implemented natively and borrow from
# existing implementations. This will be removed in the future.
add_entrypoint_object(
  acos
  SRCS
    acos.cpp
  HDRS
    ../acos.h
  VENDOR
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
)

add_entrypoint_object(
  acosh
  SRCS
    acosh.cpp
  HDRS
    ../acosh.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  asin
  SRCS
    asin.cpp
  HDRS
    ../asin.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  asinh
  SRCS
    asinh.cpp
  HDRS
    ../asinh.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  atan
  SRCS
    atan.cpp
  HDRS
    ../atan.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  atanh
  SRCS
    atanh.cpp
  HDRS
    ../atanh.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  cosh
  SRCS
    cosh.cpp
  HDRS
    ../cosh.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  erf
  SRCS
    erf.cpp
  HDRS
    ../erf.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  powi
  SRCS
    powi.cpp
  HDRS
    ../powi.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  powif
  SRCS
    powif.cpp
  HDRS
    ../powif.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  sinh
  SRCS
    sinh.cpp
  HDRS
    ../sinh.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  tanh
  SRCS
    tanh.cpp
  HDRS
    ../tanh.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  tgamma
  SRCS
    tgamma.cpp
  HDRS
    ../tgamma.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  tgammaf
  SRCS
    tgammaf.cpp
  HDRS
    ../tgammaf.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  lgamma
  SRCS
    lgamma.cpp
  HDRS
    ../lgamma.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)

add_entrypoint_object(
  lgamma_r
  SRCS
    lgamma_r.cpp
  HDRS
    ../lgamma_r.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
    -O2
  VENDOR
)
