add_header_library(
  mbstate
  HDRS
    mbstate.h
  DEPENDS
    libc.hdr.types.char32_t    
)

add_header_library(
  character_converter
  HDRS
    character_converter.h
  DEPENDS
    libc.hdr.types.char8_t
    libc.hdr.types.char32_t
    .mbstate
    .utf_ret
)

