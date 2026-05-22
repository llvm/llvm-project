add_header_library(common_constants HDRS common_constants.h)

add_header_library(
  cosf
  HDRS
    cosf.h
  DEPENDS
    libc.src.__support.CPP.simd
    libc.src.__support.math.cosf
)

add_header_library(
  exp2f
  HDRS
    exp2f.h
  DEPENDS
    libc.src.__support.CPP.simd
    libc.src.__support.math.exp2f
)

add_header_library(
  exp2m1f
  HDRS
    exp2m1f.h
  DEPENDS
    libc.src.__support.CPP.simd
    libc.src.__support.math.exp2m1f
)

add_header_library(
  exp10f
  HDRS
    exp10f.h
  DEPENDS
    libc.src.__support.CPP.simd
    libc.src.__support.math.exp10f
)

add_header_library(
  exp10m1f
  HDRS
    exp10m1f.h
  DEPENDS
    libc.src.__support.CPP.simd
    libc.src.__support.math.exp10m1f
)

add_header_library(
  expf_utils
  HDRS
    expf_utils.h
  DEPENDS
    .common_constants
    libc.src.__support.CPP.simd
)

add_header_library(
  expf
  HDRS
    expf.h
  DEPENDS
    .expf_utils
    libc.src.__support.CPP.simd
    libc.src.__support.FPUtil.fp_bits
    libc.src.__support.common
)


add_header_library(
  expm1f
  HDRS
    expm1f.h
  DEPENDS
    libc.src.__support.CPP.simd
    libc.src.__support.math.expm1f
)

add_header_library(
  sinf
  HDRS
    sinf.h
  DEPENDS
    libc.src.__support.CPP.simd
    libc.src.__support.math.sinf
)

add_header_library(
  tanf
  HDRS
    tanf.h
  DEPENDS
    libc.src.__support.CPP.simd
    libc.src.__support.math.tanf
)