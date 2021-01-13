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
  isascii
  SRCS
    isascii.cpp
  HDRS
    isascii.h
)

add_entrypoint_object(
  isblank
  SRCS
    isblank.cpp
  HDRS
    isblank.h
)

add_entrypoint_object(
  iscntrl
  SRCS
    iscntrl.cpp
  HDRS
    iscntrl.h
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

add_entrypoint_object(
  isgraph
  SRCS
    isgraph.cpp
  HDRS
    isgraph.h
  DEPENDS
    .ctype_utils
)

add_entrypoint_object(
  islower
  SRCS
    islower.cpp
  HDRS
    islower.h
  DEPENDS
    .ctype_utils
)

add_entrypoint_object(
  isprint
  SRCS
    isprint.cpp
  HDRS
    isprint.h
)

add_entrypoint_object(
  ispunct
  SRCS
    ispunct.cpp
  HDRS
    ispunct.h
  DEPENDS
    .ctype_utils
)

add_entrypoint_object(
  isspace
  SRCS
    isspace.cpp
  HDRS
    isspace.h
)

add_entrypoint_object(
  isupper
  SRCS
    isupper.cpp
  HDRS
    isupper.h
  DEPENDS
    .ctype_utils
)

add_entrypoint_object(
  isxdigit
  SRCS
    isxdigit.cpp
  HDRS
    isxdigit.h
  DEPENDS
    .ctype_utils
)

add_entrypoint_object(
  tolower
  SRCS
    tolower.cpp
  HDRS
    tolower.h
  DEPENDS
    .ctype_utils
)

add_entrypoint_object(
  toascii
  SRCS
    toascii.cpp
  HDRS
    toascii.h
)

add_entrypoint_object(
  toupper
  SRCS
    toupper.cpp
  HDRS
    toupper.h
  DEPENDS
    .ctype_utils
)
