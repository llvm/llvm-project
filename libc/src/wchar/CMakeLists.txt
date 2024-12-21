
add_entrypoint_object(
  wctob
  SRCS
    wctob.cpp
  HDRS
    wctob.h
  DEPENDS
    libc.hdr.types.wint_t
    libc.hdr.stdio_macros
    libc.src.__support.wctype_utils
)

add_entrypoint_object(
  btowc
  SRCS
    btowc.cpp
  HDRS
    btowc.h
  DEPENDS
    libc.hdr.types.wint_t
    libc.hdr.wchar_macros
    libc.src.__support.wctype_utils
)
