
add_entrypoint_object(
  wctob
  SRCS
    wctob.cpp
  HDRS
    wctob.h
  DEPENDS
    libc.include.stdio
    libc.include.wchar
    libc.src.__support.wctype_utils
)
