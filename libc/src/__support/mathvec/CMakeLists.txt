add_header_library(
  common_constants
  HDRS
    common_constants.h
  DEPENDS
)

add_header_library(
  expf_utils
  HDRS
    expf_utils.h
  DEPENDS
  libc.src.__support.CPP.simd
  libc.src.__support.mathvec.common_constants
)

add_header_library(
  expf
  HDRS
    expf.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.CPP.simd
    libc.src.__support.FPUtil.FPBits
    libc.src.__support.mathvec.expf_utils
    libc.src.__support.mathvec.vector_utils
)

add_header_library(
  vector_utils
  HDRS
    vector_utils.h
  DEPENDS
    libc.src.__support.CPP.simd
)
