add_header_library(
  app_h
  HDRS
    app.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.common
)
