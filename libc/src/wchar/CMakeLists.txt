
add_entrypoint_object(
  wctob
  SRCS
    wctob.cpp
  HDRS
    wctob.h
  DEPENDS
    libc.hdr.types.wint_t
    libc.src.__support.wctype_utils
)
