add_entrypoint_object(
  ceil
  SRCS
    ceil.cpp
  HDRS
    ../ceil.h
)

add_entrypoint_object(
  ceilf
  SRCS
    ceilf.cpp
  HDRS
    ../ceilf.h
)

add_entrypoint_object(
  copysign
  SRCS
    copysign.cpp
  HDRS
    ../copysign.h
)

add_entrypoint_object(
  copysignf
  SRCS
    copysignf.cpp
  HDRS
    ../copysignf.h
)

add_entrypoint_object(
  fabs
  SRCS
    fabs.cpp
  HDRS
    ../fabs.h
)

add_entrypoint_object(
  fabsf
  SRCS
    fabsf.cpp
  HDRS
    ../fabsf.h
)

add_entrypoint_object(
  floor
  SRCS
    floor.cpp
  HDRS
    ../floor.h
)

add_entrypoint_object(
  floorf
  SRCS
    floorf.cpp
  HDRS
    ../floorf.h
)

add_entrypoint_object(
  fma
  SRCS
    fma.cpp
  HDRS
    ../fma.h
)

add_entrypoint_object(
  fmaf
  SRCS
    fmaf.cpp
  HDRS
    ../fmaf.h
)

add_entrypoint_object(
  fmax
  SRCS
    fmax.cpp
  HDRS
    ../fmax.h
)

add_entrypoint_object(
  fmaxf
  SRCS
    fmaxf.cpp
  HDRS
    ../fmaxf.h
)

add_entrypoint_object(
  fmin
  SRCS
    fmin.cpp
  HDRS
    ../fmin.h
)

add_entrypoint_object(
  fminf
  SRCS
    fminf.cpp
  HDRS
    ../fminf.h
)

add_entrypoint_object(
  fmod
  SRCS
    fmod.cpp
  HDRS
    ../fmod.h
)

add_entrypoint_object(
  fmodf
  SRCS
    fmodf.cpp
  HDRS
    ../fmodf.h
)

add_entrypoint_object(
  nearbyint
  SRCS
    nearbyint.cpp
  HDRS
    ../nearbyint.h
)

add_entrypoint_object(
  nearbyintf
  SRCS
    nearbyintf.cpp
  HDRS
    ../nearbyintf.h
)

add_entrypoint_object(
  remainder
  SRCS
    remainder.cpp
  HDRS
    ../remainder.h
)

add_entrypoint_object(
  remainderf
  SRCS
    remainderf.cpp
  HDRS
    ../remainderf.h
)

add_entrypoint_object(
  rint
  SRCS
    rint.cpp
  HDRS
    ../rint.h
)

add_entrypoint_object(
  rintf
  SRCS
    rintf.cpp
  HDRS
    ../rintf.h
)

add_entrypoint_object(
  round
  SRCS
    round.cpp
  HDRS
    ../round.h
)

add_entrypoint_object(
  sqrt
  SRCS
    sqrt.cpp
  HDRS
    ../sqrt.h
)

add_entrypoint_object(
  sqrtf
  SRCS
    sqrtf.cpp
  HDRS
    ../sqrtf.h
)

add_entrypoint_object(
  trunc
  SRCS
    trunc.cpp
  HDRS
    ../trunc.h
)

add_entrypoint_object(
  truncf
  SRCS
    truncf.cpp
  HDRS
    ../truncf.h
)

add_entrypoint_object(
  frexp
  SRCS
    frexp.cpp
  HDRS
    ../frexp.h
)

add_entrypoint_object(
  frexpf
  SRCS
    frexpf.cpp
  HDRS
    ../frexpf.h
)

add_entrypoint_object(
  scalbn
  SRCS
    scalbn.cpp
  HDRS
    ../scalbn.h
)

add_entrypoint_object(
  scalbnf
  SRCS
    scalbnf.cpp
  HDRS
    ../scalbnf.h
)

add_entrypoint_object(
  ldexp
  SRCS
    ldexp.cpp
  HDRS
    ../ldexp.h
)

add_entrypoint_object(
  ldexpf
  SRCS
    ldexpf.cpp
  HDRS
    ../ldexpf.h
)

add_entrypoint_object(
  tgamma
  SRCS
    tgamma.cpp
  HDRS
    ../tgamma.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
)

add_entrypoint_object(
  tgammaf
  SRCS
    tgammaf.cpp
  HDRS
    ../tgammaf.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
)

add_entrypoint_object(
  lgamma
  SRCS
    lgamma.cpp
  HDRS
    ../lgamma.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
)

add_entrypoint_object(
  lgamma_r
  SRCS
    lgamma_r.cpp
  HDRS
    ../lgamma_r.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
)
