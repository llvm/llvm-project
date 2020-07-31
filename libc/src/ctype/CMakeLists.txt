add_header_library(
  ctype_utils
  HDRS
    ctype_utils.h
)

add_entrypoint_object(
  isalnum
  SRCS
    isalnum.cpp
  HDRS
    isalnum.h
  DEPENDS
    .ctype_utils
)

add_entrypoint_object(
  isalpha
  SRCS
    isalpha.cpp
  HDRS
    isalpha.h
  DEPENDS
    .ctype_utils
)

add_entrypoint_object(
  isdigit
  SRCS
    isdigit.cpp
  HDRS
    isdigit.h
  DEPENDS
    .ctype_utils
)
