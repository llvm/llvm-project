if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  add_subdirectory(${LIBC_TARGET_ARCHITECTURE})
endif()

# Implementations shared by the AMDGPU and NVPTX targets. Architecture specific
# implementations live in the corresponding subdirectory and take precedence
# over the ones listed here.
if(NOT LIBC_TARGET_ARCHITECTURE_IS_AMDGPU AND
    NOT LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  return()
endif()

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
