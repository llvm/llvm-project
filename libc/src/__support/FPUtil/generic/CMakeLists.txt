add_header_library(
  sqrt
  HDRS
    sqrt.h
    sqrt_80_bit_long_double.h
  DEPENDS
    libc.src.__support.uint128
)

add_header_library(
  fma
  HDRS
    FMA.h
  DEPENDS
    libc.src.__support.uint128
)

add_header_library(
  fmod
  HDRS
    FMod.h
)
