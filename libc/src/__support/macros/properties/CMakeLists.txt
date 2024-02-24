add_header_library(
  architectures
  HDRS
    architectures.h
)

add_header_library(
  compiler
  HDRS
    compiler.h
)

add_header_library(
  os
  HDRS
    os.h
)

add_header_library(
  cpu_features
  HDRS
    cpu_features.h
  DEPENDS
    .architectures
)

add_header_library(
  float
  HDRS
    float.h
  DEPENDS
    .architectures
    .compiler
    .cpu_features
    .os
    libc.include.llvm-libc-macros.float_macros
    libc.include.llvm-libc-types.float128
)
