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
  cpu_features
  HDRS
    cpu_features.h
  DEPENDS
    .architectures
)
