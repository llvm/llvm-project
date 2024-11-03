add_header_library(
  array
  HDRS
    array.h
)

add_header_library(
  bit
  HDRS
    bit.h
  DEPENDS
    libc.src.__support.macros.properties.compiler
)

add_header_library(
  bitset
  HDRS
    bitset.h
)

add_header_library(
  cstddef
  HDRS
    cstddef.h
  DEPENDS
    .type_traits
)

add_header_library(
  functional
  HDRS
    functional.h
)

add_header_library(
  limits
  HDRS
    limits.h
)

add_header_library(
  span
  HDRS
    span.h
  DEPENDS
    .array
    .type_traits
)

add_header_library(
  string_view
  HDRS
    string_view.h
  DEPENDS
    libc.src.__support.common
)

add_header_library(
  stringstream
  HDRS
    stringstream.h
  DEPENDS
    .span
    .string_view
    libc.src.__support.integer_to_string
)

add_header_library(
  optional
  HDRS
    optional.h
)

add_header_library(
  type_traits
  HDRS
    type_traits.h
)

add_header_library(
  utility
  HDRS
    utility.h
)

add_header_library(
  atomic
  HDRS
    atomic.h
)

add_header_library(
  expected
  HDRS
    expected.h
)

add_object_library(
  new
  SRCS
    new.cpp
  HDRS
    new.h
  DEPENDS
    libc.include.stdlib
    libc.src.__support.common
)
