add_header_library(
  common
  HDRS
    common.h
  DEPENDS
    libc.src.__support.CPP.simd
)

add_entrypoint_object(
  expf
  SRCS
    expf.cpp
  HDRS
    ../expf.h
  DEPENDS
    .common
    libc.src.__support.common
    libc.src.__support.CPP.simd
    libc.src.__support.mathvec.common_constants

)
