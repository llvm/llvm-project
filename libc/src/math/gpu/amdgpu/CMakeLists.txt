add_entrypoint_object(
  fmod
  SRCS
    fmod.cpp
  HDRS
    ../../fmod.h
)

add_entrypoint_object(
  fmodf
  SRCS
    fmodf.cpp
  HDRS
    ../../fmodf.h
)

add_entrypoint_object(
  frexp
  SRCS
    frexp.cpp
  HDRS
    ../../frexp.h
)

add_entrypoint_object(
  frexpf
  SRCS
    frexpf.cpp
  HDRS
    ../../frexpf.h
)

add_entrypoint_object(
  scalbn
  SRCS
    scalbn.cpp
  HDRS
    ../../scalbn.h
)

add_entrypoint_object(
  scalbnf
  SRCS
    scalbnf.cpp
  HDRS
    ../../scalbnf.h
)

add_entrypoint_object(
  ldexp
  SRCS
    ldexp.cpp
  HDRS
    ../../ldexp.h
)

add_entrypoint_object(
  ldexpf
  SRCS
    ldexpf.cpp
  HDRS
    ../../ldexpf.h
)

add_entrypoint_object(
  tgamma
  SRCS
    tgamma.cpp
  HDRS
    ../../tgamma.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
)

add_entrypoint_object(
  tgammaf
  SRCS
    tgammaf.cpp
  HDRS
    ../../tgammaf.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
)

add_entrypoint_object(
  lgamma
  SRCS
    lgamma.cpp
  HDRS
    ../../lgamma.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
)

add_entrypoint_object(
  lgamma_r
  SRCS
    lgamma_r.cpp
  HDRS
    ../../lgamma_r.h
  COMPILE_OPTIONS
    ${bitcode_link_flags}
)
