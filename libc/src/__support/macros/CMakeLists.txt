add_subdirectory(properties)

add_header_library(
  attributes
  HDRS
    attributes.h
)

add_header_library(
  optimization
  HDRS
    optimization.h
  DEPENDS
    libc.src.__support.macros.properties.compiler
)

add_header_library(
  sanitizer
  HDRS
    sanitizer.h
  DEPENDS
    libc.src.__support.macros.properties.compiler
)
