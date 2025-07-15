add_entrypoint_object(
  isalnum
  SRCS
    isalnum.cpp
  HDRS
    isalnum.h
  DEPENDS
    libc.include.ctype
    libc.src.__support.ctype_utils
)

add_entrypoint_object(
  isalpha
  SRCS
    isalpha.cpp
  HDRS
    isalpha.h
  DEPENDS
    libc.src.__support.ctype_utils
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
    libc.src.__support.ctype_utils
)

add_entrypoint_object(
  isgraph
  SRCS
    isgraph.cpp
  HDRS
    isgraph.h
  DEPENDS
    libc.src.__support.ctype_utils
)

add_entrypoint_object(
  islower
  SRCS
    islower.cpp
  HDRS
    islower.h
  DEPENDS
    libc.src.__support.ctype_utils
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
    libc.src.__support.ctype_utils
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
    libc.src.__support.ctype_utils
)

add_entrypoint_object(
  isxdigit
  SRCS
    isxdigit.cpp
  HDRS
    isxdigit.h
  DEPENDS
    libc.src.__support.ctype_utils
)

add_entrypoint_object(
  tolower
  SRCS
    tolower.cpp
  HDRS
    tolower.h
  DEPENDS
    libc.src.__support.ctype_utils
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
    libc.src.__support.ctype_utils
)

# Do not build the locale versions in overlay mode.
if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

add_entrypoint_object(
  isalnum_l
  SRCS
    isalnum_l.cpp
  HDRS
    isalnum_l.h
  DEPENDS
    libc.include.ctype
    libc.src.__support.ctype_utils
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  isalpha_l
  SRCS
    isalpha_l.cpp
  HDRS
    isalpha_l.h
  DEPENDS
    libc.src.__support.ctype_utils
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  isblank_l
  SRCS
    isblank_l.cpp
  HDRS
    isblank_l.h
  DEPENDS
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  iscntrl_l
  SRCS
    iscntrl_l.cpp
  HDRS
    iscntrl_l.h
  DEPENDS
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  isdigit_l
  SRCS
    isdigit_l.cpp
  HDRS
    isdigit_l.h
  DEPENDS
    libc.src.__support.ctype_utils
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  isgraph_l
  SRCS
    isgraph_l.cpp
  HDRS
    isgraph_l.h
  DEPENDS
    libc.src.__support.ctype_utils
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  islower_l
  SRCS
    islower_l.cpp
  HDRS
    islower_l.h
  DEPENDS
    libc.src.__support.ctype_utils
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  isprint_l
  SRCS
    isprint_l.cpp
  HDRS
    isprint_l.h
  DEPENDS
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  ispunct_l
  SRCS
    ispunct_l.cpp
  HDRS
    ispunct_l.h
  DEPENDS
    libc.src.__support.ctype_utils
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  isspace_l
  SRCS
    isspace_l.cpp
  HDRS
    isspace_l.h
  DEPENDS
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  isupper_l
  SRCS
    isupper_l.cpp
  HDRS
    isupper_l.h
  DEPENDS
    libc.src.__support.ctype_utils
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  isxdigit_l
  SRCS
    isxdigit_l.cpp
  HDRS
    isxdigit_l.h
  DEPENDS
    libc.src.__support.ctype_utils
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  tolower_l
  SRCS
    tolower_l.cpp
  HDRS
    tolower_l.h
  DEPENDS
    libc.src.__support.ctype_utils
    libc.hdr.types.locale_t
)

add_entrypoint_object(
  toupper_l
  SRCS
    toupper_l.cpp
  HDRS
    toupper_l.h
  DEPENDS
    libc.src.__support.ctype_utils
    libc.hdr.types.locale_t
)
