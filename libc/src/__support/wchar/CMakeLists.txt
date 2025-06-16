add_header_library(
  mbstate
  HDRS
    mbstate.h
  DEPENDS
    libc.hdr.types.char32_t    
)

add_object_library(
  character_converter
  HDRS
    character_converter.h
  SRCS 
    character_converter.cpp
  DEPENDS
    libc.hdr.types.char8_t
    libc.hdr.types.char32_t
    .mbstate
    .utf_ret
)

add_header_library(
  utf_ret
  HDRS
    utf_ret.h
)
